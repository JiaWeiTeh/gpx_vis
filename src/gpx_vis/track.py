"""Core of gpx_vis: load ``.gpx`` tracks and draw them on an interactive map.

A :class:`Track` reads one ``.gpx`` file or a folder of them, keeps the raw
latitude / longitude / elevation / time arrays, derives distance, speed, stops
and moving time from them, and can write a self-contained HTML map (built with
folium) with one route per track, coloured by speed (default) or elevation.
"""
import logging
import math
import operator
import os
from collections import Counter, defaultdict
from datetime import timedelta

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

# A stop = staying within STOP_RADIUS_M of one spot for at least STOP_MIN_S
# seconds. GPS positions keep wandering while you stand still; inside a stop no
# distance is added, speed is zero and the time does not count as moving time.
STOP_RADIUS_M = 15.0
STOP_MIN_S = 60.0

_M_PER_DEG = 111_195.0  # metres per degree of latitude (mean Earth radius)

# What create_map can colour the routes by -> label used for the legend.
_COLOR_BY = {
    'speed': 'Speed (km/h)',
    'elevation': 'Elevation (m)',
}

# Popup chart colours (light popup surface): one hue per panel, ink for text.
_SPEED_COLOR = '#2a78d6'
_ELEVATION_COLOR = '#eb6834'
_INK = '#0b0b0b'
_INK_SECONDARY = '#52514e'
_INK_MUTED = '#898781'
_GRID = '#e1e0d9'
_AXIS = '#c3c2b7'
_CHART_WIDTH = 320
_POPUP_SIZE = (400, 385)  # iframe width, height in px

# Dark outline under each route so light colours stay visible on light maps.
_CASING = dict(color='#1a1a19', weight=7, opacity=0.45)

# Background layers offered in the layer control (bottom-right corner of the map).
# CARTO basemaps officially require a free API key; see ``Track.create_map``.
_CARTO_LAYERS = [
    ('Positron', 'Plain'),
    ('DarkMatter', 'Dark mode'),
    ('Voyager', 'Plain (hierarchical)'),
]


def _fmt_duration(td):
    """Compact duration: '45 s', '12 min', '3 h 05 min'."""
    seconds = td.total_seconds()
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes = int(round(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min" if hours else f"{minutes} min"


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
        self._track_list = []
        self._segment_list = []
        self._n_tracks = 0
        self._n_segments = 0
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
        # which <trk> and which <trkseg> every point belongs to (unique across files)
        self._track_id = np.array(self._track_list, dtype=int)
        self._segment_id = np.array(self._segment_list, dtype=int)
        # free the temporary lists
        del (self._x_list, self._y_list, self._z_list, self._t_list, self._name_list,
             self._track_list, self._segment_list)
        # warn about missing data
        if any(t is None for t in self.t):
            logger.warning(
                "Some GPS points have no timestamp. "
                "Speed, duration and create_map() need timestamps on every point."
            )
        if np.isnan(self.z).any():
            logger.warning(
                "Some GPS points have no elevation; they are left out of elevation colours and charts."
            )
        # caches for derived quantities
        self._data_cache = None
        self._seconds_cache = None
        self._stop_cache = None
        self._step_cache = None
        self._speed_cache = None
        _timer.end()

    def _record(self, gpx):
        """
        Going through gpx.tracks.segments.points and appending all values.
        """
        for trk in gpx.tracks:
            track_id = self._n_tracks
            self._n_tracks += 1
            for sgmt in trk.segments:
                segment_id = self._n_segments
                self._n_segments += 1
                for pt in sgmt.points:
                    self._y_list.append(pt.latitude)
                    self._x_list.append(pt.longitude)
                    self._z_list.append(pt.elevation if pt.elevation is not None else np.nan)
                    self._t_list.append(pt.time)
                    self._name_list.append(trk.name)
                    self._track_list.append(track_id)
                    self._segment_list.append(segment_id)

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

    def help(self):
        """Print the project URL."""
        logger.info('Check out https://github.com/JiaWeiTeh/gpx_vis .')

    # =============================================================================
    # Distance, stops, speed and moving time
    # =============================================================================

    @property
    def speed(self) -> np.ndarray:
        """
        Ground speed at every point in km/h (NaN where timestamps are missing).

        GPS positions jitter by a few metres, so speed is smoothed: at each point
        it is the distance travelled divided by the time taken over a window of
        SPEED_WINDOW_S seconds centred on that point. Computed separately for each
        GPS segment; zero inside stops.
        """
        if self._speed_cache is None:
            self._speed_cache = self._compute_speed()
        return self._speed_cache

    @property
    def summary(self) -> pd.DataFrame:
        """
        One row per route: start/end time (UTC), distance, total duration,
        moving time (stops and recording gaps left out), average speed over the
        total and over the moving time, and top speed.
        """
        return pd.DataFrame([self._stats(start, end) for start, end in self.idx_trksplit()])

    @property
    def _seconds(self) -> np.ndarray:
        """Unix time in seconds for every point (NaN where the timestamp is missing)."""
        if self._seconds_cache is None:
            times = pd.to_datetime(pd.Series(self.t, dtype=object), utc=True)
            self._seconds_cache = (times - pd.Timestamp(0, tz='UTC')).dt.total_seconds().to_numpy()
        return self._seconds_cache

    @property
    def _stopped(self) -> np.ndarray:
        """
        True for points inside a stop: a run of points that stays within
        STOP_RADIUS_M of its first point for at least STOP_MIN_S seconds.
        """
        if self._stop_cache is None:
            stopped = np.zeros(len(self.x), dtype=bool)
            seconds = self._seconds
            for start, end in self._segments():
                t = seconds[start:end].tolist()
                n = end - start
                if n < 2 or np.isnan(t).any():
                    continue
                # flat local projection in metres: accurate at the 15 m scale
                coslat = math.cos(math.radians(self.y[start]))
                north = ((self.y[start:end] - self.y[start]) * _M_PER_DEG).tolist()
                east = ((self.x[start:end] - self.x[start]) * _M_PER_DEG * coslat).tolist()
                i = 0
                while i < n - 1:
                    j = i
                    while j + 1 < n and math.hypot(east[j + 1] - east[i], north[j + 1] - north[i]) <= STOP_RADIUS_M:
                        j += 1
                    if j > i and t[j] - t[i] >= STOP_MIN_S:
                        stopped[start + i:start + j + 1] = True
                        i = j + 1
                    else:
                        i += 1
            self._stop_cache = stopped
        return self._stop_cache

    @property
    def _step_km(self) -> np.ndarray:
        """
        Distance in km from the previous point. Zero at the start of every GPS
        segment (gaps are not bridged) and between two points of a stop.
        """
        if self._step_cache is None:
            steps = np.zeros(len(self.x))
            stopped = self._stopped
            for start, end in self._segments():
                for i in range(start + 1, end):
                    if stopped[i] and stopped[i - 1]:
                        continue
                    steps[i] = vincenty((self.y[i - 1], self.x[i - 1]), (self.y[i], self.x[i]))
            self._step_cache = steps
        return self._step_cache

    def _compute_speed(self) -> np.ndarray:
        seconds = self._seconds
        steps = self._step_km
        speed = np.full(len(self.x), np.nan)
        half_window = SPEED_WINDOW_S / 2
        for start, end in self._segments():
            t = seconds[start:end]
            if end - start < 2 or np.isnan(t).any():
                continue
            # guard against out-of-order timestamps
            t = np.maximum.accumulate(t)
            if t[-1] <= t[0]:
                continue
            dist = np.cumsum(steps[start:end])  # km since segment start
            # np.interp needs strictly increasing times: keep the last of any duplicates
            keep = np.r_[t[1:] > t[:-1], True]
            lo = np.clip(t - half_window, t[0], t[-1])
            hi = np.clip(t + half_window, t[0], t[-1])
            covered = np.interp(hi, t[keep], dist[keep]) - np.interp(lo, t[keep], dist[keep])
            speed[start:end] = covered / (hi - lo) * 3600.0
        # one-point segments: borrow from neighbours in the same track
        for start, end in self.idx_trksplit():
            v = speed[start:end]
            if np.isnan(v).any() and np.isfinite(v).any():
                speed[start:end] = pd.Series(v).interpolate(limit_direction='both').to_numpy()
        return speed

    def _time_weights(self) -> np.ndarray:
        """Seconds each point stands for: half the gap to its neighbours in the same segment."""
        weights = np.zeros(len(self.x))
        seconds = self._seconds
        for start, end in self._segments():
            t = seconds[start:end]
            if end - start < 2 or np.isnan(t).any():
                continue
            half = np.diff(t) / 2
            weights[start:end - 1] += half
            weights[start + 1:end] += half
        return weights

    def _stats(self, start, end) -> dict:
        """Distance, duration, moving time and speeds of the points [start, end)."""
        seconds = self._seconds[start:end]
        distance_km = float(self._step_km[start + 1:end].sum())
        stats = {
            'name': self.name[start],
            'start': self.t[start],
            'end': self.t[end - 1],
            'distance_km': distance_km,
            'duration': None,
            'moving_time': None,
            'avg_speed_kmh': np.nan,
            'moving_speed_kmh': np.nan,
            'max_speed_kmh': np.nan,
        }
        if np.isnan(seconds).any():
            return stats
        # moving: time between consecutive points of one segment, unless both are in a stop
        segment = self._segment_id[start:end]
        stopped = self._stopped[start:end]
        moving = (segment[1:] == segment[:-1]) & ~(stopped[1:] & stopped[:-1])
        duration_s = float(seconds[-1] - seconds[0])
        moving_s = float(np.diff(seconds)[moving].sum())
        speed = self.speed[start:end]
        stats.update(
            duration=timedelta(seconds=round(duration_s)),
            moving_time=timedelta(seconds=round(moving_s)),
            avg_speed_kmh=distance_km / (duration_s / 3600) if duration_s > 0 else np.nan,
            moving_speed_kmh=distance_km / (moving_s / 3600) if moving_s > 0 else np.nan,
            max_speed_kmh=float(np.nanmax(speed)) if np.isfinite(speed).any() else np.nan,
        )
        return stats

    # =============================================================================
    # Here we deal with cities we have been in the tour.
    # =============================================================================

    class City:
        """
        Class that handles city information from a dictionary. For example:
        >>> city.city = Neckargemünd
        >>> city.country = Germany
        >>> city.code = DE
        >>> city.frequency = 204      # number of track points
        >>> city.time = timedelta(minutes=17)  # time spent there
        """

        def __init__(self, city_data):
            self.city = city_data['city']
            self.country = city_data['country']
            self.code = city_data['country_code']
            self.frequency = 0  # placeholder. Will be calculated.
            self.time = timedelta(0)  # placeholder. Will be calculated.
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
            time = humanfriendly.format_timespan(self.time, max_units=2)
            return f"{{country: {self.country}, city: {self.city}, frequency: {self.frequency}, time: {time}}}\n"
        # add tuple sorting system. We want to sort by country first, then by city.
        def __lt__(self, other):
            return (self.country, self.city) < (other.country, other.city)

    @property
    def city_list(self) -> list:
        """
        Cities passed through, sorted by country then name. Each has `frequency`
        (number of track points there) and `time` (time spent there, which stays
        right when the recording interval changes along the way).
        """
        # batch reverse geocode all coordinates at once (much faster than per-point)
        all_results = reverse_geocode.search(list(zip(self.y, self.x)))
        cities = [self.City(result) for result in all_results]
        counts = Counter(cities)
        seconds = defaultdict(float)
        for city, weight in zip(cities, self._time_weights()):
            seconds[city] += weight
        unique_city_list = []
        for city, count in counts.items():
            city.frequency = count
            city.time = timedelta(seconds=round(seconds[city]))
            unique_city_list.append(city)
        # return full list of cities, sorted by country then by name
        logger.info('Here are the cities you passed through on your journey.')
        return sorted(unique_city_list)

    # =============================================================================
    # Track handling
    # =============================================================================

    @staticmethod
    def _runs(ids, start=0, end=None):
        """[start, end) index pairs of runs of equal values in ids[start:end]."""
        ids = ids[start:end]
        if len(ids) == 0:
            return []
        cuts = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        bounds = np.r_[0, cuts, len(ids)] + start
        return [[int(a), int(b)] for a, b in zip(bounds[:-1], bounds[1:])]

    def idx_trksplit(self):
        """
        [start, end) index pairs, one per track: every <trk> of every file is its
        own route, even when two of them share a name.
        """
        return self._runs(self._track_id)

    def _segments(self, start=0, end=None):
        """[start, end) index pairs, one per GPS segment (<trkseg>)."""
        return self._runs(self._segment_id, start, end)

    def _sample(self, start, end, max_points, keep_segment_ends=True):
        """
        Indices of about `max_points` evenly spaced points of [start, end), always
        including the first and last point (and each segment's ends).
        """
        if end - start <= max_points:
            return np.arange(start, end)
        idx = np.round(np.linspace(start, end - 1, max_points)).astype(int)
        if keep_segment_ends:
            ends = [i for a, b in self._segments(start, end) for i in (a, b - 1)]
            idx = np.r_[idx, ends]
        return np.unique(idx)

    # =============================================================================
    # Plotting on maps
    # =============================================================================

    def create_map(self, filename, lite=False, nlite=50, color_by='speed', carto_api_key=None):
        """
        Draw every track on an interactive map and save it as an HTML file.

        filename: output path (str or pathlib.Path); '.html' is appended if missing.
        lite: if True, draw at most about `nlite` points per track (faster for big files).
        nlite: point budget per track in lite mode (minimum 10).
        color_by: 'speed' (default) or 'elevation': what the route colours show
            (one colour scale for the whole map, shown in a legend). Each route's
            popup chart always shows speed and elevation.
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
        for selected_idx in self.idx_trksplit():
            self._add_tracks_on_map(line_group, selected_idx, lite, nlite, values, colormap)

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
                width = 200, height = 150,
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

    def _add_tracks_on_map(self, group, selected_idx, lite, nlite, values, colormap):
        """
        This function adds individual tracks onto create_map().
        group: FeatureGroup this track belongs to.
        selected_idx: index range of this particular track.
        lite, nlite: see create_map().
        values, colormap: per-point quantity the route is coloured by, and its scale.
        """
        start_idx, end_idx = selected_idx
        idx = self._sample(start_idx, end_idx, nlite) if lite else np.arange(start_idx, end_idx)
        # fill unknown values (e.g. missing elevation) from neighbours for colouring only
        point_values = (pd.Series(values[idx]).interpolate(limit_direction='both')
                        .fillna(colormap.vmin).to_numpy())
        segment_of = self._segment_id[idx]
        tooltip = self._add_tooltip(selected_idx)

        # one line per GPS segment, so recording gaps are not bridged
        lines = []
        for segment in pd.unique(segment_of):
            mask = segment_of == segment
            coords = list(zip(self.y[idx][mask], self.x[idx][mask]))
            if len(coords) < 2:
                continue
            seg_values = point_values[mask]
            folium.PolyLine(coords, **_CASING).add_to(group)
            # colour each piece by the mean of its two end points
            folium.ColorLine(coords,
                            colors = (seg_values[:-1] + seg_values[1:]) / 2,
                            colormap = colormap,
                            weight = 4,
                            ).add_to(group)
            lines.append(coords)

        # start marker and finish flag at the true first and last point
        popup = folium.Popup(self._add_popup_chart(selected_idx), max_width=_POPUP_SIZE[0] + 20)
        folium.CircleMarker(location = (self.y[start_idx], self.x[start_idx]),
                            radius = 5,
                            fill = True,
                            color = 'black',
                            stroke = True,
                            fill_opacity = 1,
                            fill_color = 'yellow',
                            ).add_to(group)
        folium.Marker(location = (self.y[end_idx - 1], self.x[end_idx - 1]),
                      icon=folium.Icon(color="green", icon="flag"),
                      popup = popup,
                      ).add_to(group)
        if not lines:
            return

        # wide transparent layer on top: hover highlight and tooltip
        def highlight_function(_feature):
            return {'color': '#8fe60e', 'opacity': .5, 'weight': 10}

        def style_function(_feature):
            return {'color': 'transparent', 'weight': 25}

        # a full Feature with properties: folium's generated JS reads feature.properties.style
        # and breaks the whole map when given a bare geometry
        folium.features.GeoJson(
                data = {
                    'type': 'Feature',
                    'properties': {},
                    'geometry': {
                        'type': 'MultiLineString',
                        # GeoJSON wants (lon, lat)
                        'coordinates': [[[lon, lat] for lat, lon in line] for line in lines],
                    },
                },
                control=False,
                tooltip = tooltip,
                style_function=style_function,
                highlight_function=highlight_function,
                ).add_to(group)

    def _add_popup_txt(self, selected_idx):
        """
        Title and summary lines for a route's popup.
        """
        start_idx, end_idx = selected_idx
        stats = self._stats(start_idx, end_idx)
        start_city = self.City(reverse_geocode.search([(self.y[start_idx], self.x[start_idx])])[0]).city
        end_city = self.City(reverse_geocode.search([(self.y[end_idx - 1], self.x[end_idx - 1])])[0]).city

        t0, t1 = stats['start'], stats['end']
        title = f"{stats['name']}"
        if t0.date() == t1.date():
            subtitle1 = f"{t0:%d.%m.%Y}, {t0:%H:%M} to {t1:%H:%M} UTC"
        else:
            subtitle1 = f"{t0:%d.%m.%Y %H:%M} to {t1:%d.%m.%Y %H:%M} UTC"
        subtitle2 = f"{start_city} to {end_city}"
        subtitle3 = (f"{stats['distance_km']:.2f} km in {_fmt_duration(stats['duration'])},"
                     f" moving {_fmt_duration(stats['moving_time'])}")
        if np.isfinite(stats['moving_speed_kmh']):
            subtitle4 = (f"Moving average {stats['moving_speed_kmh']:.1f} km/h,"
                         f" top speed {stats['max_speed_kmh']:.1f} km/h")
        else:
            subtitle4 = 'Speed: n/a'
        return title, subtitle1, subtitle2, subtitle3, subtitle4

    def _popup_chart(self, selected_idx) -> alt.TopLevelMixin:
        """
        Speed over time, with elevation in a second panel below on the same time
        axis (two panels rather than two y-axes on one plot, whose alignment would
        be arbitrary). Hovering shows a crosshair and the values at that time.
        """
        start_idx, end_idx = selected_idx
        idx = self._sample(start_idx, end_idx, 200, keep_segment_ends=False)
        title, *subtitles = self._add_popup_txt(selected_idx)
        chart_data = pd.DataFrame({
            'time': pd.to_datetime(pd.Series(self.t[idx], dtype=object), utc=True),
            'speed': np.round(self.speed[idx], 1),
            'elevation': np.round(self.z[idx]),
            'distance': np.round(np.cumsum(self._step_km[start_idx:end_idx])[idx - start_idx], 2),
        })
        has_elevation = bool(np.isfinite(chart_data['elevation']).any())

        hover = alt.selection_point(fields=['time'], nearest=True, on='pointerover',
                                    clear='pointerout', empty=False)
        tooltip = [
            alt.Tooltip('utchoursminutesseconds(time):T', title='Time (UTC)'),
            alt.Tooltip('speed:Q', title='Speed (km/h)', format='.1f'),
            alt.Tooltip('distance:Q', title='Distance (km)', format='.2f'),
        ]
        if has_elevation:
            tooltip.insert(2, alt.Tooltip('elevation:Q', title='Elevation (m)', format='.0f'))

        def panel(field, label, color, height, scale, area, show_time_axis):
            base = alt.Chart(chart_data).encode(
                x=alt.X('time:T', scale=alt.Scale(type='utc'),
                        axis=alt.Axis(format='%H:%M', tickCount=5, title='Time (UTC)')
                        if show_time_axis else alt.Axis(labels=False, ticks=False, title=None)),
            )
            y = alt.Y(f'{field}:Q', title=label, scale=scale, axis=alt.Axis(tickCount=4))
            if area:
                # clip: the fill runs down to zero, which may sit below the panel
                mark = base.mark_area(color=color, opacity=0.1, clip=True,
                                      line={'color': color, 'strokeWidth': 2})
            else:
                mark = base.mark_line(color=color, strokeWidth=2, strokeJoin='round', strokeCap='round')
            # crosshair: invisible rules at every point; the nearest one shows on hover
            rule = base.mark_rule(color=_INK_MUTED, strokeWidth=1).encode(
                opacity=alt.condition(hover, alt.value(1), alt.value(0)),
                tooltip=tooltip,
            ).add_params(hover)
            dot = base.mark_point(filled=True, size=64, color=color, stroke='white', strokeWidth=2).encode(
                y=y, opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            )
            return alt.layer(mark.encode(y=y), rule, dot).properties(width=_CHART_WIDTH, height=height)

        top_speed = float(np.nanmax(chart_data['speed'])) if np.isfinite(chart_data['speed']).any() else 1.0
        speed_panel = panel('speed', 'Speed (km/h)', _SPEED_COLOR, 150 if has_elevation else 220,
                            alt.Scale(domain=[0, max(1.0, top_speed * 1.1)], nice=True),
                            area=False, show_time_axis=not has_elevation)
        if has_elevation:
            elevation = chart_data['elevation'].dropna()
            pad = max(10.0, 0.1 * (elevation.max() - elevation.min()))
            elevation_panel = panel('elevation', 'Elevation (m)', _ELEVATION_COLOR, 90,
                                    alt.Scale(domain=[float(elevation.min() - pad), float(elevation.max() + pad)],
                                              nice=False),
                                    area=True, show_time_axis=True)
            chart = alt.vconcat(speed_panel, elevation_panel, spacing=10)
        else:
            chart = speed_panel
        return (
            chart.properties(title=alt.Title(title, subtitle=subtitles, anchor='start', limit=_POPUP_SIZE[0] - 30))
            .configure_axis(labelColor=_INK_MUTED, titleColor=_INK_SECONDARY, titleFontWeight='normal',
                            gridColor=_GRID, domainColor=_AXIS, tickColor=_AXIS,
                            labelFontSize=11, titleFontSize=11)
            .configure_view(stroke=None)
            .configure_title(color=_INK, subtitleColor=_INK_SECONDARY, fontSize=14, subtitleFontSize=11)
        )

    def _add_popup_chart(self, selected_idx):
        """
        The popup chart as an iframe holding Altair's own HTML page, which loads
        the Vega versions that match the installed Altair. (folium.VegaLite picks
        script versions from a fixed list and loads Vega-Lite 2 for Altair 6
        charts, which then fail to draw.)
        """
        html = self._popup_chart(selected_idx).to_html(embed_options={'actions': False})
        html = html.replace('<head>', '<head><style>body { margin: 4px; font-family: system-ui, sans-serif; }</style>', 1)
        return folium.IFrame(html, width=_POPUP_SIZE[0], height=_POPUP_SIZE[1])

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
