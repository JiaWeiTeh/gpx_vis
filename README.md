### About `gpx_vis`Transforms your .GPX files into an interactive map, whether you're conquering mountain trails or cruising city streets.Whether you're using apps like <a href="https://www.komoot.com/" target="_blank">Komoot</a> and <a href="https://www.strava.com/" target="_blank">Strava</a>, or crafting tracks on platforms like <a href="https://gpx.studio/" target="_blank">gpx.studio</a> - just make sure your  file includes data for `latitude`, `longitude`, `elevation`, and `time`, and you're good to go!Important note: `gpx_vis` is still in its development phase, so expect more exciting features and enhancements down the trail. But in the meantime, buckle up and get ready to explore your adventures! :climbing::bicyclist:### Contents[Installation](#Install)<br>[How to use](#Instantiate)<br>* [Map creation](#Map-creation)* [City overview](#City-overview)* [Additional information](#Additional-information)## InstallInstalling `gpx_vis` is straightforward through <a href="https://pypi.org/project/gpx-vis/" target="_blank">PyPI</a>:```bashpip install gpx_vis```To update to the latest version (if applicable):```bashpip install gpx_vis --upgrade```## Instantiate```pythonfrom gpx_vis import Tracktrack = Track(pathname)````pathname` serves as a reference to either a single `.gpx` file or a directory.If a directory path is provided, all .gpx files contained within that directorywill be merged, but retain their original metadata (e.g., different tracks remain separated).This allows users to overlay multiple tracks onto a single map, providing a comprehensive summary of the data being analyzed.## Map creation```pythontrack.create_map(filename)```* This generates an HTML file saved under `filename`, which contains an interactive map <a href="https://raw.githack.com/JiaWeiTeh/gpx_vis/main/example/map.html" target="_blank">(example)</a>.* Each segment of a track corresponds to the elevation relative to user's position: blue = lower elevation, yellow = higher elevation. * Users can hover over routes or click on final waypoints (flags) for additional information:![](https://raw.githubusercontent.com/JiaWeiTeh/gpx_vis/main/media/track_info.png)Different backgrounds are available (selectable at lower-right corner):|Tile type             |  Example  ||----------------------|-----------||Dark |  ![](https://raw.githubusercontent.com/JiaWeiTeh/gpx_vis/main/media/ex_dark.png) ||Street view | ![](https://raw.githubusercontent.com/JiaWeiTeh/gpx_vis/main/media/ex_openstreet.png) ||Plain |  ![](https://raw.githubusercontent.com/JiaWeiTeh/gpx_vis/main/media/ex_plain.png) ||Plain (Heirarchical) |  ![](https://raw.githubusercontent.com/JiaWeiTeh/gpx_vis/main/media/ex_plainheir.png) |In cases where the map contains a large dataset and loading time is a concern (if onehas been very hardworking), users have the option to include `lite = True`. * By default, this option limits the number of data points displayed on each route to `nlite = 50`. This is adjustable using `nlite` argument:```pythontrack.create_map(filename, lite=True, nlite=100)```Note that the minimum value for `nlite` is `10`.## City overview```pythontrack.city_list```Interested in the cities you passed through on your journey? This command generates a listof cities - sorted by country and name - that were detected along the tour route. * The `frequency` parameter indicates how many times each city was encountered, providing a rough estimate of the time spent in each city.## Additional informationIf you wish to plot or inspect different values, these parameters are also accessible via:* `track.t`: time (UTC)<br>* `track.x`: longitude (deg) - x values on a graph would represent the latitude.<br> * `track.y`: latitude (deg)<br>* `track.z`: elevation (m)<br>For a more comprehensive and unprocessed dataset, you can use:`track.data`:  Returns all data in a `pandas.DataFrame` accessible format.## Dependencies(numpy) https://numpy.org/ <br>(pandas) https://pandas.pydata.org/ <br>(altair) https://altair-viz.github.io/ <br>(branca) https://pypi.org/project/branca/ <br>(gpxpy) https://github.com/tkrajina/gpxpy <br>(vincenty) https://pypi.org/project/vincenty/ <br>(folium) https://github.com/python-visualization/folium <br>(humanfriendly) https://pypi.org/project/humanfriendly/ <br>(reverse-geocode) https://pypi.org/project/reverse-geocode/ <br>