#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 19:44:24 2024

@author: Jia Wei Teh
"""

from gpx_vis import Track

pathname = r'./example/data/'
# load track
track = Track(pathname)
# show cities
print(track.city_list)
# show interactive map
track.create_map(r'./example/map.html')