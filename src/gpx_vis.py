#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 2 23:14:31 2024

@author: Jia Wei Teh

This script combines .gpx files in /data and overplots them onto a HTML file.
"""
# main library
import gpxpy

import os
import math
import logging
import branca
import folium
import numpy as np
import pandas as pd
import altair as alt
import humanfriendly
import reverse_geocode

from time import time
from vincenty import vincenty
from datetime import timedelta
from folium.plugins import MarkerCluster, MiniMap

logger = logging.getLogger(__name__)

class Track:
    """
    Instance used to process .gpx files.
    """
    
    # =============================================================================
    # Initialisation
    # =============================================================================
    
    def __init__(self, pathname: str):
        """
        Open .gpx file and set values.
        pathname: str to either a gpx file, or a directory containing them (thus merging them).
        """
        if not isinstance(pathname, str):
            raise TypeError(f"'pathname' must be a string, got {type(pathname).__name__}.")
        # Intermediate lists for collecting values (converted to arrays after).
        # Note: y = latitude, x = longitude.
        self._x_list = []
        self._y_list = []
        self._t_list = []
        self._z_list = []
        self._name_list = []
        _timer = Timer()
        _timer.begin('Reading data...')
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
                "Time-dependent features (duration, elevation graph) may not work correctly."
            )
        # cache for the data property
        self._data_cache = None
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
        # some column name here. TBD cause headers not finalised.
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
        from collections import Counter
        city_list = []
        # find nearest city from coords via reverse_geocode.
        for coords in zip(self.y, self.x):
            city = self.City(reverse_geocode.search([coords])[0])
            city_list.append(city)
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
        Note: x -> x[i,j], x[k+1, l]. See plt_tracks().
        """
        idx_list =  np.where(self.name[:-1] != self.name[1:])[0]
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
    # Plotting on graphs    
    # =============================================================================

        
    @staticmethod
    def _round2n(x, n):
        """rounds to n significant numbers"""
        return round(x, -int(math.floor(np.log10(x))) + (n - 1))
        
    
    
    # =============================================================================
    # Plotting on maps
    # =============================================================================
    
    def create_map(self, filename: str, lite: bool = False, **kwargs) -> None:
        """
        Map out your tour on an interactive streetmaps.
        """
        # start timer
        _timer = Timer()
        _timer.begin()
        logger.info('Mapping data...')
        # find optimal center for map display.
        map_center = self.data[['latitude', 'longitude']].mean().values.tolist()
        # southwest (minimums) and northeast (maximums) boundary.
        map_sw = self.data[['latitude', 'longitude']].min().values.tolist()
        map_ne = self.data[['latitude', 'longitude']].max().values.tolist()
        # create Map.
        main_map = folium.Map(location = map_center)
        # specify border.
        main_map.fit_bounds([map_sw, map_ne])
        
        # create group
        line_group = folium.FeatureGroup(name = "Your Routes")
        # plot waypoints for each end and beginning of a track
        idx_split_list = self.idx_trksplit()
        # if list is empty, there is no splitting tracks
        for selected_idx in idx_split_list:
            self._add_tracks_on_map(line_group, selected_idx, lite, **kwargs)

        # clusters
        cluster = MarkerCluster().add_to(main_map)
        # add group to map
        line_group.add_to(cluster)

        # add different backgrounds
        tiles_list = ['cartodbpositron', 'Cartodb dark_matter', 'CartoDB Voyager' ]
        tile_names = ['Plain', 'Dark mode', 'Plain (hierarchical)']
        for i, tiles in enumerate(tiles_list):
            folium.raster_layers.TileLayer(tiles, name = tile_names[i]).add_to(main_map)
        # add layer control
        folium.LayerControl(position='bottomright').add_to(main_map)
        # add minimap 
        MiniMap(toggle_display = True, zoom_level_offset = -4,
                width = 400, height = 200,
                position = 'topright',
                ).add_to(main_map)
        
        # save
        if not filename.endswith(".html"):
            filename += '.html'
        main_map.save(filename)
        # show time
        _timer.end()
        logger.info(f"File saved as {filename}.")
    
    def _add_tracks_on_map(self, group, selected_idx, lite, **kwargs):
        """
        This function adds individual tracks onto create_map().
        group: FeatureGroup this track belongs to.
        selected_idx: index range of this particular track.
        """
        start_idx, end_idx = selected_idx
        # validate inputs
        if not isinstance(lite, bool):
            raise TypeError(f"'lite' must be True or False, got {type(lite).__name__}.")
        if kwargs.get('nlite') is not None:
            if kwargs.get('nlite') < 10:
                raise ValueError("Minimum value of 'nlite' is 10.")
            max_n_points = kwargs.get('nlite')
        else:
            max_n_points = 50
        # calculate the actual index interval for desired points
        if (end_idx - start_idx) < max_n_points or not lite:
            n_points = 1
        else:
            n_points = int((end_idx - start_idx) / max_n_points)
        
            
        track_coords = list(zip(self.y[start_idx:end_idx:n_points], self.x[start_idx:end_idx:n_points]))
        elevation_graph = self._add_popup_graph(selected_idx)
        # create popup
        popup = folium.Popup(min_width=400,
                             max_width=400)
        elevation_graph.add_to(popup)
        # add tooltip
        tooltip = self._add_tooltip(selected_idx)
        # add to group
        # since elevation sometimes differ wildly, perhaps it is better to use
        # log-scale as a simple fix. (as long as there aren't zero entries)
        # Right now, I am using individual tracks for individual colorbar min/max. Can of course
        # switch to map-wide colorbar by removing [start_idx:end_idx]
        folium.ColorLine(track_coords,
                        colors = self.z[start_idx:end_idx:n_points],
                        colormap = branca.colormap.linear.plasma.scale(min(self.z[start_idx:end_idx:n_points]),max(self.z[start_idx:end_idx:n_points])),
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
        dist = self._get_distance(track_y, track_x)
        time_elapsed = self._get_time_elapsed(track_t[0], track_t[-1])

        title = f'{track_name}'
        subtitle1 = f"""Start: {track_t[0].strftime('%d.%m.%Y %H:%M:%S')} (UTC), {start_city}"""
        subtitle2 = f"""End: {track_t[-1].strftime('%d.%m.%Y %H:%M:%S')} (UTC), {end_city}"""
        subtitle3 = f'Total: {dist} km, {time_elapsed}'

        return title, subtitle1, subtitle2, subtitle3
    
    def _add_popup_graph(self, selected_idx):
        """
        Creates elevation graph in Popup text.
        """
        start_idx, end_idx = selected_idx
        # limit number of points
        max_n_points = 100
        if (end_idx - start_idx) < max_n_points:
            n_points = 1
        else:
            n_points = int((end_idx - start_idx) / max_n_points)

        # titles
        title, subtitle1, subtitle2, subtitle3 = self._add_popup_txt(selected_idx)
        # plot
        lineplot = (
            alt.Chart(
                self.data[['time', 'elevation']][start_idx:end_idx:n_points],
                title=alt.Title(title, subtitle=[subtitle1, subtitle2, subtitle3])
            )
            .mark_line()
            .encode(
                x=alt.X('time:T', axis=alt.Axis(tickCount=6)).title('Time (Local)'),
                y=alt.Y('elevation:Q', axis=alt.Axis(tickMinStep=20))
                    .scale(domain=(
                        min(self.z[start_idx:end_idx:n_points] - 50),
                        max(self.z[start_idx:end_idx:n_points] + 50)
                    ))
                    .title('Elevation (m)'),
            )
            .properties(width=300, height=300)
        )

        # turn into vega
        elevation_graph = folium.VegaLite(
                            lineplot,
                            width=300,
                            height=300,
                            )
        return elevation_graph
    
    
    def _add_tooltip(self, selected_idx):
        """
        Creates str-block that contains tooltip when mouse is hovered over the track.
        """
        start_idx, _end_idx = selected_idx
        track_name = self.name[start_idx]
        return f"Route: {track_name}"
    
    @staticmethod
    def _get_distance(latlist, lonlist):
        """
        Distance travelled in kilometers, by adding up bits of routes.
        """
        lat1s = latlist[:-1]
        lat2s = latlist[1:]

        lon1s = lonlist[:-1]
        lon2s = lonlist[1:]

        return round(np.cumsum([vincenty((lat1, lon1), (lat2, lon2)) for lat1, lat2, lon1, lon2 in zip(lat1s, lat2s, lon1s, lon2s)])[-1], 3)

    @staticmethod
    def _get_time_elapsed(start, end):
        """
        Calculate time elapsed.
        """
        elapsed = end - start
        return humanfriendly.format_timespan(elapsed)
    
    @property
    def shouldiContinueCycling(self):
        logger.info('yes of course.')
    
    
    
# =============================================================================
# A script that calculates time elapsed, for debugging and performance purposes.
# =============================================================================

class Timer:
    """
    Timer class that calculates time elapsed and prints it out
        in a human-friendly way. Based on .datetime and .humanfriendly.
    Uses: 1) from clock import timer
          2) _timer.begin('optional str here')
          3) _timer.end()
          4) profit
    """
    
    # initialisation
    def __init__(self):
        # start time
        self.start = None
        # end time
        self.stop = None
    
    # converts time elapsed into string
    def secs2str(self):
        # calculate difference
        elapsed = timedelta(seconds = self.stop - self.start)
        # return in formatted string
        return humanfriendly.format_timespan(elapsed)

    # sets beginning of timer
    def begin(self, s = ''):
        # record the beginning time
        self.start = time()

    # sets end of timer
    def end(self):
        # make sure start is invoked:
        if self.start is None:
            raise InvalidTimerCall('_timer.end() called, but .begin() not detected.')
        # record the end time
        self.stop = time()
        # then, print out time elapsed.
        time_str = self.secs2str()
        separator = '~' * (len(time_str) + 15)
        logger.debug(f'{separator}\nTime elapsed: {time_str}.\n{separator}')
        # reset
        self.start = None
        self.stop = None

class InvalidTimerCall(Exception):
    """
    Raised when timer call is invalid. 

    For example: calling _timer.end() without explicitly calling _timer.begin().
    """
