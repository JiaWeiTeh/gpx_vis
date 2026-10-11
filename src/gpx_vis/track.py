"""Core of gpx_vis: load ``.gpx`` tracks and draw them on an interactive map.

A :class:`Track` reads one ``.gpx`` file or a folder of them, keeps the raw
latitude / longitude / elevation / time arrays, and can write a self-contained
HTML map (built with folium) with one route per track, coloured by speed
(default) or elevation.
"""
import logging
import operator
import os
from collections import Counter

import altair as alt
import branca
import folium
import gpxpy
import humanfriendly
import numpy as np
import pandas as pd
import reverse_geocode
import xyzservices.providers as xyz
from folium.plugins import MarkerCluster, MiniMap
from vincenty import vincenty

from .timer import Timer

logger = logging.getLogger(__name__)

# Speed at a point = distance travelled / time taken over a window of this many
# seconds centred on the point. Raw point-to-point GPS speeds are far too noisy
# (a few metres of position jitter per second is several km/h of fake speed).
SPEED_WINDOW_S = 30.0

# What create_map can colour the routes by -> label used for the legend and chart axis.
_COLOR_BY = {
    'speed': 'Speed (km/h)',
    'elevation': 'Elevation (m)',
}

# Background layers offered in the layer control (bottom-right corner of the map).
# CARTO basemaps officially require a free API key; see ``Track.create_map``.
_CARTO_LAYERS = [
    ('Positron', 'Plain'),
    ('DarkMatter', 'Dark mode'),
    ('Voyager', 'Plain (hierarchical)'),
]


class Track:
    """
    Instance used to process .gpx files.
    """

    # =============================================================================
    # Initialisation
    # =============================================================================

    def __init__(self, pathname):
        """
        Open .gpx file and set values.
        pathname: path (str or pathlib.Path) to either a gpx file, or a directory
                  containing them (thus merging them).
        """
        if not isinstance(pathname, (str, os.PathLike)):
            raise TypeError(f"'pathname' must be a str or Path, got {type(pathname).__name__}.")
        pathname = os.fspath(pathname)
        # Intermediate lists for collecting values (converted to arrays after).
        # Note: y = latitude, x = longitude.
        self._x_list = []
        self._y_list = []
        self._t_list = []
        self._z_list = []
        self._name_list = []
        _timer = Timer()
        _timer.begin()
        logger.info('Reading data...')
        # validate that the path exists
        if not os.path.exists(pathname):
            raise FileNotFoundError(f"Path not found: '{pathname}'")
        # if pathname is a folder
        # loop through file.
        if os.path.isdir(pathname):
            for fname in sorted(os.listdir(pathname)):
                if fname.endswith('.gpx'):
                    with open(os.path.join(pathname, fname), 'r') as file:
                        gpx = gpxpy.parse(file)
                    # record values
                    self._record(gpx)
        # else just read
        elif os.path.isfile(pathname):
            if pathname.endswith('.gpx'):
                with open(pathname, 'r') as file:
                    gpx = gpxpy.parse(file)
                # record values
                self._record(gpx)
        # check that we actually parsed some GPS points
        if len(self._x_list) == 0:
            raise ValueError(
                f"No GPS points found. Ensure '{pathname}' contains valid .gpx files with track data."
            )
        # convert collected lists to numpy arrays
        self.x = np.array(self._x_list, dtype=float)
        self.y = np.array(self._y_list, dtype=float)
        self.z = np.array(self._z_list, dtype=float)
        self.t = np.array(self._t_list, dtype=object)
        self.name = np.array(self._name_list, dtype=object)
        # free the temporary lists
        del self._x_list, self._y_list, self._z_list, self._t_list, self._name_list
        # warn about missing data
        if any(t is None for t in self.t):
            logger.warning(
                "Some GPS points have no timestamp. "
                "Speed, duration and create_map() need timestamps on every point."
            )
        # caches for derived quantities
        self._data_cache = None
        self._step_cache = None
        self._speed_cache = None
        _timer.end()

    def _record(self, gpx):
        """
        Going through gpx.tracks.segments.points and appending all values.
        """
        for trk in gpx.tracks:
            for sgmt in trk.segments:
                for pt in sgmt.points:
                    self._y_list.append(pt.latitude)
                    self._x_list.append(pt.longitude)
                    self._z_list.append(pt.elevation if pt.elevation is not None else 0.0)
                    self._t_list.append(pt.time)
                    self._name_list.append(trk.name)

    @property
    def header(self):
        """Column names of ``Track.data``."""
        return self.data.columns.values

    @property
    def data(self) -> pd.DataFrame:
        """
        Shows pd.DataFrame object from input gpx file.
        """
        if self._data_cache is not None:
            return self._data_cache
        col_names = ['trackName', 'latitude', 'longitude', 'elevation', 'time']
        data = { col_names[0]: self.name,
                col_names[1]: self.y,
                col_names[2]: self.x,
                col_names[3]: self.z,
                col_names[4]: self.t,
                }
        self._data_cache = pd.DataFrame(data=data)
        return self._data_cache

    @property
    def speed(self) -> np.ndarray:
        """
        Ground speed at every point in km/h (NaN where timestamps are missing).

        GPS positions jitter by a few metres, so speed is smoothed: at each point
        it is the distance travelled divided by the time taken over a window of
        SPEED_WINDOW_S seconds centred on that point (always including at least
        one neighbour on each side). Computed separately for each track.
        """
        if self._speed_cache is None:
            self._speed_cache = self._compute_speed()
        return self._speed_cache

    @property
    def _step_km(self) -> np.ndarray:
        """Distance in km from the previous point of the same track (0 at track starts)."""
        if self._step_cache is None:
            steps = np.zeros(len(self.x))
            for start_idx, end_idx in self.idx_trksplit():
                for i in range(start_idx + 1, end_idx):
                    steps[i] = vincenty((self.y[i - 1], self.x[i - 1]), (self.y[i], self.x[i]))
            self._step_cache = steps
        return self._step_cache

    def _seconds(self) -> np.ndarray:
        """Unix time in seconds for every point (NaN where the timestamp is missing)."""
        times = pd.to_datetime(pd.Series(self.t, dtype=object), utc=True)
        return (times - pd.Timestamp(0, tz='UTC')).dt.total_seconds().to_numpy()

    def _compute_speed(self) -> np.ndarray:
        seconds = self._seconds()
        steps = self._step_km
        speed = np.full(len(self.x), np.nan)
        half_window = SPEED_WINDOW_S / 2
        for start_idx, end_idx in self.idx_trksplit():
            t = seconds[start_idx:end_idx]
            n = len(t)
            if n < 2 or np.isnan(t).any():
                continue
            # guard against out-of-order timestamps so the search below is valid
            t = np.maximum.accumulate(t)
            dist = np.cumsum(steps[start_idx:end_idx])  # km since track start
            idx = np.arange(n)
            lo = np.minimum(np.searchsorted(t, t - half_window, side='left'), np.maximum(idx - 1, 0))
            hi = np.maximum(np.searchsorted(t, t + half_window, side='right') - 1, np.minimum(idx + 1, n - 1))
            dt = t[hi] - t[lo]
            with np.errstate(divide='ignore', invalid='ignore'):
                v = np.where(dt > 0, (dist[hi] - dist[lo]) / dt * 3600.0, np.nan)
            # windows with no time progress (duplicate timestamps): borrow from neighbours
            speed[start_idx:end_idx] = pd.Series(v).interpolate(limit_direction='both').to_numpy()
        return speed

    def help(self):
        """Print the project URL."""
        logger.info('Check out https://github.com/JiaWeiTeh/gpx_vis .')

    # =============================================================================
    # Here we deal with cities we have been in the tour.
    # =============================================================================

    class City:
        """
        Class that handles city information from a dictionary. For example:
        >>> city.city = Neckargemünd
        >>> city.country = Germany
        >>> city.code = DE
        >>> city.frequency = 204
        """

        def __init__(self, city_data):
            self.city = city_data['city']
            self.country = city_data['country']
            self.code = city_data['country_code']
            self.frequency = 0 #placeholder. Will be calculated.
        # equivalency and hashing for set().
        def __eq__(self, other):
            if isinstance(other, self.__class__):
                if self.city == other.city and self.country == other.country and self.code == other.code:
                    return True
            return False
        def __hash__(self):
            return hash((self.city, self.country, self.code))
        # to be unambiguous for info purposes. Return as string.
        def __repr__(self) -> str:
            return f"{{country: {self.country}, city: {self.city}, frequency: {self.frequency}}}\n"
        # add tuple sorting system. We want to sort by country first, then by city.
        def __lt__(self, other):
            return (self.country, self.city) < (other.country, other.city)

    @property
    def city_list(self) -> list:
        """
        Obtain information of cities visited during the tour (including duplicates).
        """
        # batch reverse geocode all coordinates at once (much faster than per-point)
        all_coords = list(zip(self.y, self.x))
        all_results = reverse_geocode.search(all_coords)
        city_list = [self.City(result) for result in all_results]
        # count frequencies in O(n) using Counter
        city_counts = Counter(city_list)
        unique_city_list = []
        for city, count in city_counts.items():
            city.frequency = count
            unique_city_list.append(city)
        # return full list of cities, sorted by country then by name
        logger.info('Here are the cities you passed through on your journey.')
        return sorted(unique_city_list)

    # =============================================================================
    # Track handling
    # =============================================================================

    def idx_trksplit(self):
        """
        Index at which we enter a new track entry (if any).
        Returns a list of [start, end) index pairs, one per track.
        """
        idx_list = np.where(self.name[:-1] != self.name[1:])[0]
        # we provide list of indices at which tracks separate.
        track_list = []
        # list is empty if there is only one track route.
        if len(idx_list) == 0:
            track_list.append([0, len(self.name)])
            return track_list
        else:
            # record index from previous loop
            previous_idx = 0
            # corner case
            if len(idx_list) == 1:
                idx = idx_list[0]
                track_list.append([0, idx+1])
                track_list.append([idx+1, len(self.name)])
            else:
                for ii, idx in enumerate(idx_list):
                    # start value
                    if ii == 0:
                        track_list.append([0, idx+1])
                        previous_idx = idx + 1
                    # end value
                    elif ii == (len(idx_list) - 1):
                        # account for both cases in the last loop
                        track_list.append([previous_idx, idx + 1])
                        track_list.append([idx + 1, len(self.name)])
                    # in-between values
                    else:
                        track_list.append([previous_idx, idx + 1])
                        previous_idx = idx + 1
            return track_list

    # =============================================================================
    # Plotting on maps
    # =============================================================================

    def create_map(self, filename, lite=False, nlite=50, color_by='speed', carto_api_key=None):
        """
        Draw every track on an interactive map and save it as an HTML file.

        filename: output path (str or pathlib.Path); '.html' is appended if missing.
        lite: if True, draw at most about `nlite` points per track (faster for big files).
        nlite: point budget per track in lite mode (minimum 10).
        color_by: 'speed' (default) or 'elevation'. Sets both the route colours
            (one colour scale for the whole map, shown in a legend) and the
            chart in each route's popup.
        carto_api_key: key for the CARTO backgrounds (Plain, Dark mode,
            Plain (hierarchical)). Defaults to the CARTO_API_KEY environment
            variable. Without a key those layers may stay blank; the
            OpenStreetMap and Topographic backgrounds need no key.
            Note: the key is written into the generated HTML file.
        Returns the path of the written file.
        """
        # validate inputs
        if not isinstance(lite, bool):
            raise TypeError(f"'lite' must be True or False, got {type(lite).__name__}.")
        try:
            nlite = operator.index(nlite)
        except TypeError:
            raise TypeError(f"'nlite' must be an integer, got {type(nlite).__name__}.") from None
        if nlite < 10:
            raise ValueError("Minimum value of 'nlite' is 10.")
        if color_by not in _COLOR_BY:
            raise ValueError(f"'color_by' must be one of {sorted(_COLOR_BY)}, got {color_by!r}.")
        if any(t is None for t in self.t):
            raise ValueError(
                "Some GPS points have no timestamp. gpx_vis needs time data for the "
                "duration, speed and the popup chart."
            )
        filename = os.fspath(filename)
        if not filename.endswith('.html'):
            filename += '.html'
        # start timer
        _timer = Timer()
        _timer.begin()
        logger.info('Mapping data...')
        # find optimal center for map display.
        map_center = self.data[['latitude', 'longitude']].mean().values.tolist()
        # southwest (minimums) and northeast (maximums) boundary.
        map_sw = self.data[['latitude', 'longitude']].min().values.tolist()
        map_ne = self.data[['latitude', 'longitude']].max().values.tolist()
        # create Map (default background: OpenStreetMap).
        main_map = folium.Map(location = map_center)
        # specify border.
        main_map.fit_bounds([map_sw, map_ne])

        # create group
        line_group = folium.FeatureGroup(name = "Your Routes")
        # one colour scale for the whole map, so routes can be compared; shown as a legend
        values = self.speed if color_by == 'speed' else self.z
        colormap = self._colormap(values, _COLOR_BY[color_by])
        colormap.add_to(main_map)
        # plot waypoints for each end and beginning of a track
        idx_split_list = self.idx_trksplit()
        # if list is empty, there is no splitting tracks
        for selected_idx in idx_split_list:
            self._add_tracks_on_map(line_group, selected_idx, lite, nlite, values, colormap, color_by)

        # clusters
        cluster = MarkerCluster().add_to(main_map)
        # add group to map
        line_group.add_to(cluster)

        # add different backgrounds
        key = carto_api_key or os.environ.get('CARTO_API_KEY')
        for provider_name, label in _CARTO_LAYERS:
            provider = getattr(xyz.CartoDB, provider_name)
            tiles = provider(apikey=key) if key else provider
            folium.TileLayer(tiles=tiles, name=label).add_to(main_map)
        folium.TileLayer(tiles='OpenTopoMap', name='Topographic').add_to(main_map)
        # add layer control
        folium.LayerControl(position='bottomright').add_to(main_map)
        # add minimap
        MiniMap(toggle_display = True, zoom_level_offset = -4,
                width = 400, height = 200,
                position = 'topright',
                ).add_to(main_map)

        # save
        main_map.save(filename)
        # show time
        _timer.end()
        logger.info(f"File saved as {filename}.")
        return filename

    @staticmethod
    def _colormap(values, caption):
        """
        Plasma colour scale (purple = low, yellow = high) spanning the 2nd to 98th
        percentile of `values`, so a few GPS glitches don't wash out the colours.
        Values outside the range get the end colours.
        """
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = np.floor(np.percentile(finite, 2)), np.ceil(np.percentile(finite, 98))
            if vmax <= vmin:
                vmin, vmax = vmin - 1, vmax + 1
        colormap = branca.colormap.linear.plasma.scale(vmin, vmax)
        colormap.caption = caption
        return colormap

    def _add_tracks_on_map(self, group, selected_idx, lite, nlite, values, colormap, color_by):
        """
        This function adds individual tracks onto create_map().
        group: FeatureGroup this track belongs to.
        selected_idx: index range of this particular track.
        lite, nlite: see create_map().
        values, colormap: per-point quantity the route is coloured by, and its scale.
        """
        start_idx, end_idx = selected_idx
        # calculate the actual index interval for desired points
        if lite and (end_idx - start_idx) >= nlite:
            n_points = int((end_idx - start_idx) / nlite)
        else:
            n_points = 1

        track_coords = list(zip(self.y[start_idx:end_idx:n_points], self.x[start_idx:end_idx:n_points]))
        popup_graph = self._add_popup_graph(selected_idx, color_by)
        # create popup
        popup = folium.Popup(min_width=400,
                             max_width=400)
        popup_graph.add_to(popup)
        # add tooltip
        tooltip = self._add_tooltip(selected_idx)
        # colour each segment by the mean of its two end points
        point_values = values[start_idx:end_idx:n_points]
        point_values = np.where(np.isfinite(point_values), point_values, colormap.vmin)
        segment_values = (point_values[:-1] + point_values[1:]) / 2
        folium.ColorLine(track_coords,
                        colors = segment_values,
                        colormap = colormap,
                        tooltip = tooltip,
                        weight = 4,
                        ).add_to(group)

        # add start/finish points
        folium.CircleMarker(location = track_coords[0],
                            radius = 5,
                            fill = True,
                            color = 'black',
                            stroke = True,
                            fill_opacity = 1,
                            fill_color = 'yellow',
                            ).add_to(group)
        folium.Marker(location = track_coords[-1],
                      icon=folium.Icon(color="green", icon="flag"),
                      popup = popup,
                      ).add_to(group)
        # add highlight functionality
        def highlight_function(_feature):
            return {'color': '#8fe60e', 'opacity': .5, 'weight': 10}

        highlight_line = {
            'type': 'LineString',
            # reverse coord from (lat, lon) into (lon, lat) for GeoJSON
            'coordinates': [coord[::-1] for coord in track_coords]
        }
        # add transparent layer to help detect highlighting
        folium.features.GeoJson(
                color = 'transparent',
                data = highlight_line,
                control=False,
                tooltip = tooltip,
                weight = 25,
                highlight_function=highlight_function,
                ).add_to(group)

    def _add_popup_txt(self, selected_idx):
        """
        Creates str-block that contains useful info.
        """
        start_idx, end_idx = selected_idx
        track_name = self.name[start_idx]
        track_y = self.y[start_idx:end_idx]
        track_x = self.x[start_idx:end_idx]
        track_t = self.t[start_idx:end_idx]
        # get information
        start_city = self.City(reverse_geocode.search([[track_y[0], track_x[0]]])[0]).city
        end_city = self.City(reverse_geocode.search([[track_y[-1], track_x[-1]]])[0]).city
        dist = round(float(self._step_km[start_idx + 1:end_idx].sum()), 3)
        time_elapsed = self._get_time_elapsed(track_t[0], track_t[-1])
        hours = (track_t[-1] - track_t[0]).total_seconds() / 3600
        track_v = self.speed[start_idx:end_idx]

        title = f'{track_name}'
        subtitle1 = f"""Start: {track_t[0].strftime('%d.%m.%Y %H:%M:%S')} (UTC), {start_city}"""
        subtitle2 = f"""End: {track_t[-1].strftime('%d.%m.%Y %H:%M:%S')} (UTC), {end_city}"""
        subtitle3 = f'Total: {dist} km, {time_elapsed}'
        if hours > 0 and np.isfinite(track_v).any():
            subtitle4 = f'Average speed: {dist / hours:.1f} km/h, max {np.nanmax(track_v):.1f} km/h'
        else:
            subtitle4 = 'Average speed: n/a'

        return title, subtitle1, subtitle2, subtitle3, subtitle4

    def _add_popup_graph(self, selected_idx, color_by='speed'):
        """
        Creates the speed (or elevation) over time chart shown in the popup.
        """
        start_idx, end_idx = selected_idx
        # limit number of points
        max_n_points = 100
        if (end_idx - start_idx) < max_n_points:
            n_points = 1
        else:
            n_points = int((end_idx - start_idx) / max_n_points)

        # titles
        title, *subtitles = self._add_popup_txt(selected_idx)
        values = self.speed if color_by == 'speed' else self.z
        chart_data = pd.DataFrame({
            'time': pd.to_datetime(pd.Series(self.t[start_idx:end_idx:n_points], dtype=object), utc=True),
            'value': values[start_idx:end_idx:n_points],
        }).dropna()
        if color_by == 'speed':
            # speed is a ratio scale: start the axis at zero
            y_scale = alt.Scale(zero=True)
        else:
            y_scale = alt.Scale(domain=(chart_data['value'].min() - 50, chart_data['value'].max() + 50))
        # plot
        lineplot = (
            alt.Chart(chart_data, title=alt.Title(title, subtitle=subtitles))
            .mark_line()
            .encode(
                # timestamps are UTC; a 'utc' scale stops the browser shifting them
                # into the viewer's own time zone
                x=alt.X('time:T', axis=alt.Axis(tickCount=6, format='%H:%M'), scale=alt.Scale(type='utc'), title='Time (UTC)'),
                y=alt.Y('value:Q', scale=y_scale, title=_COLOR_BY[color_by]),
            )
            .properties(width=300, height=300)
        )

        # turn into vega
        return folium.VegaLite(
                            lineplot,
                            width=300,
                            height=300,
                            )

    def _add_tooltip(self, selected_idx):
        """
        Creates str-block that contains tooltip when mouse is hovered over the track.
        """
        start_idx, _end_idx = selected_idx
        track_name = self.name[start_idx]
        return f"Route: {track_name}"

    @staticmethod
    def _get_time_elapsed(start, end):
        """
        Calculate time elapsed.
        """
        elapsed = end - start
        return humanfriendly.format_timespan(elapsed)

    @property
    def should_i_continue_cycling(self):
        logger.info('yes of course.')
