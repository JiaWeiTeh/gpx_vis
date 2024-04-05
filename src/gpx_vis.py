#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 2 23:14:31 2024

@author: Jia Wei Teh

This script combines .gpx files in /data and overplots them onto a HTML file?

Written as an extension based on gpxpy.

The problem with other libraries is that they dont really have outputs i want,
and are not very customisable. This hopefully solves the problem (a little).
"""
# main library
import gpxpy

import os
import math
import folium
import numpy as np
import pandas as pd
import humanfriendly
import cmasher as cmr
import reverse_geocode
import matplotlib.pyplot as plt

from vincenty import vincenty
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

class Track:
    """
    Instance used to process .gpx files.
    """
    
    # =============================================================================
    # Intialisation
    # =============================================================================
    
    def __init__(self, pathname):
        """
        Open .gpx file and set values.
        pathname: str to either a gpx file, or a directory containing them (thus merging them).
        """
        # quick-and-dirty way to record values
        # Note: latitude is the horizontal line, which corresponds to 'y' in plotting.
        #       Likewise, x = longitude.
        self.x = []
        self.y = []
        # time
        self.t = []
        # elevation
        self.z = []
        # name
        self.name = []
        
        # if pathname is a folder
        # loop through file.
        if os.path.isdir(pathname):
            for fname in os.listdir(pathname):
                if fname.endswith('.gpx'):
                    with open(os.path.join(pathname, fname), 'r') as file:
                        self.gpx = gpxpy.parse(file)
                    # record values
                    self._record()
        # else just read
        elif os.path.isfile(pathname):
            if pathname.endswith('.gpx'):
                with open(pathname, 'r') as file:
                    self.gpx = gpxpy.parse(file)
                # record values
                self._record()
        # simple file check.
        if len(self.x) == 0:
            raise FileNotFoundError('No .gpx file found in given path or directory.')
        
    def _record(self):
        """
        Going through gpx.tracks.segments.points and appending all values.
        """
        # tracks
        for trk in self.gpx.tracks:
            # segments
            for sgmt in trk.segments:
                # points
                for pt in sgmt.points:
                    # grab values
                    # Note: latitude is the horizontal line, which corresponds to 'y' in plotting.
                    self.y = np.concatenate((self.y, [pt.latitude]))
                    self.x = np.concatenate((self.x, [pt.longitude]))
                    self.z = np.concatenate((self.z, [pt.elevation]))
                    self.t = np.concatenate((self.t, [pt.time]))
                    self.name = np.concatenate((self.name, [trk.name]))
                    
    @property
    def header(self):
        # some column name here. TBD cause headers not finalised.
        return
    
    @property
    def data(self):
        """
        Shows pd.DataFrame object from input gpx file.
        """
        # some pandas library here. Set column names
        # col_names = ['trackName', 'latitude (y; deg)', 'longitude (x; deg)', 'elevation (z; m)', 'time (t; datetime)']
        col_names = ['trackName', 'latitude', 'longitude', 'elevation', 'time']
        # data
        data = { col_names[0]: self.name,
                col_names[1]: self.y,
                col_names[2]: self.x,
                col_names[3]: self.z,
                col_names[4]: self.t,
                }
        # create dataframe
        df = pd.DataFrame(data = data)
        # return
        return df
    
    # @property
    # def help(self):
    #     return "Documentations TBD"
            
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

    # property instead of method, so we do not have to call track.city_list().
    @property
    def city_list(self):
        """
        Obtain information of cities visited during the tour (including duplicates).
        """
        # initialise list of cities
        city_list = []
        # find nearest city from coords via reverse_geocode.
        for coords in zip(self.y, self.x):
            # create City instance, using dictionary output from reverse_geocode.
            city = self.City(reverse_geocode.search([coords])[0]) #[0] to remove list.
            city_list.append(city)
        # remove duplicates 
        unique_city_list = list(set(city_list))
        # add frequency of appearance of city in tour
        for ii, unique_city in enumerate(unique_city_list):
            counts = city_list.count(unique_city)
            # update attribute
            setattr(unique_city_list[ii], 'frequency', counts)
        # return full list of cities, sorted by country then by name
        return sorted(unique_city_list)
    
    # =============================================================================
    # Track handling
    # =============================================================================
    
    @staticmethod
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
            for ii, idx in enumerate(idx_list):
                # start value
                if ii == 0:
                    track_list.append([0, idx+1])
                    previous_idx = idx + 1
                # end value
                elif ii == (len(idx_list) - 1):
                    # account for both cases in the last loop
                    track_list.append([previous_idx + 1, idx + 1])
                    track_list.append([idx + 1, len(self.name)])
                # in-between values
                else:
                    track_list.append([previous_idx + 1, idx + 1])
                    previous_idx = idx + 1
            return track_list

    # =============================================================================
    # Plotting on graphs    
    # =============================================================================

    def plt_tracks(self):
        """
        Creates a latitude-longitude plot with matplotlib, coloured by elevation.
        """
        # create subplot
        fig, ax = plt.subplots(1, 1, figsize = (5, 5), dpi = 300)
        # draw line
        plt.scatter(self.x, self.y, c = self.z, cmap = cmr.rainforest,
                    vmax = self._round2n(max(self.z), 3),  
                    vmin = self._round2n(min(self.z), 3),
                    s = 0.5,)
        # colorbar
        plt.colorbar(label = 'elevation (m)')
                     
        
        # plot waypoints for each end and beginning of a track
        idx_split_list = self.idx_trksplit(self)
        for (i, j) in idx_split_list:
            self.plt_waypoints(ax, self.x[i:j], self.y[i:j])
            
        # ticks
        # 5 mini ticks between major ticks
        xtick_n, ytick_n = 5, 5
        xspan = max(self.x) - min(self.x)
        yspan = max(self.y) - min(self.y)
        # 5 ticks on each sides
        xinterval = self._round2n(xspan/5, 2)
        yinterval = self._round2n(yspan/5, 2)
        ax.xaxis.set_major_locator(MultipleLocator(xinterval))
        ax.xaxis.set_minor_locator(AutoMinorLocator(xtick_n))
        ax.yaxis.set_major_locator(MultipleLocator(yinterval))
        ax.yaxis.set_minor_locator(AutoMinorLocator(ytick_n))
        ax.tick_params(axis='both', which = 'major', direction = 'in',length = 6, width = 1)
        ax.tick_params(axis='both', which = 'minor', direction = 'in',length = 4, width = 1)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')

        plt.show()

    @staticmethod
    def plt_waypoints(ax, x, y):
        """
        Adding mini waypoints to the plt_tracks() plot.
        """
        # A small trick to separate tracks without actually separating them: add waypoint at end and start of tracks
        # start
        ax.scatter(x[0], y[0], 
                marker = '.', s = 100, 
                alpha = 0.8,
                fc = 'yellow', ec = 'k')
        # finish
        ax.scatter(x[-1], y[-1], 
                marker = '^', s = 50, 
                alpha = 0.8,
                fc = 'blue', ec = 'k')
        
    @staticmethod
    def _round2n(x, n):
        """rounds to n significant numbers"""
        return round(x, -int(math.floor(np.log10(x))) + (n - 1))
        
    
    
    # =============================================================================
    # Plotting on maps
    # =============================================================================
    
    def create_map(self, filename):
        """
        Map out your tour on an interactive streetmaps.
        """
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
        lineGroup = folium.FeatureGroup("Lines")
        
        # plot waypoints for each end and beginning of a track
        idx_split_list = self.idx_trksplit(self)
        # if list is empty, there is no splitting tracks
        for selected_idx in idx_split_list:
            self._addTracksOnMap(self, lineGroup, selected_idx)
        # add group to map
        lineGroup.add_to(main_map)
        
        # add different backgrounds
        _tilesList = ['openstreetmap', 'CartoDB Voyager', 'Cartodb dark_matter', 'cartodbpositron']
        for tiles in _tilesList:
            folium.TileLayer(tiles).add_to(main_map)
        # add layer control
        folium.LayerControl(position='bottomright').add_to(main_map)
        
        # save
        if not filename.endswith(".html"):
            filename += '.html'
        main_map.save(filename)
        
        return print(f"File saved as {filename}.")
    
    @staticmethod
    def _addTracksOnMap(self, group, selected_idx):
        """
        This function adds individual tracks onto create_map().
        group: FeatureGroup this track belongs to.
        selected_idx: index range of this particular track.
        """
        ii, jj = selected_idx
        track_coords = list(zip(self.y[ii:jj], self.x[ii:jj]))
        # popup str in html
        popupTxt = self._addPopuptxt(self, selected_idx)
        # information frame        
        iframe = folium.IFrame(popupTxt)
        # create popup
        popup = folium.Popup(iframe,
                     min_width=300,
                     max_width=300)
        # add tooltip
        tooltip = self._addTooltip(self, selected_idx)
        # add to group
        folium.PolyLine(track_coords,
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
        # 1. hover functionality.
        highlight_function = lambda x: {'color':'#8fe60e', 
                                        'opacity': .5,
                                        'weight': 10}
        # 2. highlighted line
        highlight_line = {'geometry': {
                    'type': 'LineString',
                    # reverse coord from (y, x) into (x,y)
                    'coordinates': [coord[::-1] for coord in track_coords]
                    }}
        # add
        folium.features.GeoJson(
                data = highlight_line['geometry'],
                control=False,
                tooltip = tooltip,
                highlight_function=highlight_function, 
                ).add_to(group)
        
        return  
      
    @staticmethod
    def _addPopuptxt(self, selected_idx):
        """
        Creates str-block that contains useful info about the route when clicked.
        """
        # index range
        ii, jj = selected_idx
        # track name
        track_name = self.name[ii]
        track_y = self.y[ii:jj]
        track_x = self.x[ii:jj]
        track_t = self.t[ii:jj]
        # get information
        startCity = self.City(reverse_geocode.search([[track_y[0], track_x[0]]])[0]).city
        endCity = self.City(reverse_geocode.search([[track_y[-1], track_x[-1]]])[0]).city
        dist = self._getDistance(track_y, track_x)
        timeElapsed = self._getTimeElapsed(track_t[0], track_t[-1])
        
        # HTML fmt
        infostr = f"""
                    <h3>{track_name}</h3>
                    <h4> {startCity} - {endCity}</h4>
                    <p> 
                    <b>Start</b>: <em>{track_t[0].strftime('%d.%m.%Y %H:%M:%S')}</em><br>
                    <b>End</b>: <em>{track_t[-1].strftime('%d.%m.%Y %H:%M:%S')}</em><br>
                    <b>Dist</b>: {dist} km<br>
                    <b>Duration</b>: {timeElapsed}<br>
                    </p>
                  """
        return infostr
    
    @staticmethod
    def _addTooltip(self, selected_idx):
        """
        Creates str-block that contains tooltip when mouse is hovered over the track.
        """
        ii, jj = selected_idx
        track_name = self.name[ii]
        infostr = f"Route: {track_name}"
        return infostr
    
    @staticmethod
    def _getDistance(xlist, ylist):
        """
        Distance travelled in kilometers, by adding up bits of routes.
        """
        
        x1s = xlist[:-1]
        x2s = xlist[1:]
        
        y1s = ylist[:-1]
        y2s = ylist[1:]
        
        return round(np.cumsum([vincenty((x1, y1), (x2, y2)) for x1, x2, y1, y2 in zip(x1s, x2s, y1s, y2s)])[-1], 3)
    
    @staticmethod
    def _getTimeElapsed(start, end):
        """
        Calculate time elasped.
        """
        # calculate difference
        elapsed = end - start
        # readable
        return humanfriendly.format_timespan(elapsed)
    
    @property
    def shouldiContinueCycling(self):
        return 'yes of course.'
