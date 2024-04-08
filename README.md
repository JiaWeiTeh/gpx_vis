About `gpx_vis`===============`gpx_vis` enables simple and convenient visualisation of your cycling or hiking tours.Important note: Currently, it exclusively supports .GPX files generated from completed paths in Komoot. Please be aware that this tool is still in development.Install-------```bashpip install gpx_vis```Update to the latest version:```bashpip install gpx_vis --upgrade```Instantiate-----------```pythonfrom gpx_vis import Tracktrack = Track(pathname)````pathname` serves as a reference to either a single `.gpx` file or a directory.If a directory path is provided, all .gpx files contained within that directorywill be merged, but retain their original metadata (e.g., different tracks remain separated).This allows users to overlay multiple tracks onto a single map, providing a comprehensive summary of the data being analyzed.Map creation------------```pythontrack.create_map(filename)```This generates an HTML file saved under `filename`, which contains an interactive map.Users can hover over routes or click on final waypoints (flags) for additional information.In cases where the map contains a large dataset and loading time is a concern (if onehas been very hardworking), users have the option to include `lite = True`. By default, this option limits the number of data points displayed on each route to `50`. However, users can adjust thislimit using the `nlite` argument, for instance,```pythontrack.create_map(filename, lite=True, nlite=100)```City overview-------------```pythontrack.city_list```Interested in the cities you passed through on your journey? This command generates a listof cities - sorted by country and name - that were detected along the tour route. The `frequency` parameter indicates how many times each city was encountered,providing a rough estimate of the time spent in each city.Additional information----------------------If you wish to plot or inspect different values, these parameters are also accessible via:`track.t`: time (UTC)<br>`track.x`: longitude (deg)<br>`track.y`: latitude (deg)<br>`track.z`: elevation (m)<br>For a more comprehensive and unprocessed data:`track.data`:  returns a `pandas.DataFrame` with headers ['trackName', 'latitude', 'longitude', 'elevation', 'time']`Dependencies------------(numpy) https://numpy.org/ <br>(pandas) https://pandas.pydata.org/ <br>(altair) https://altair-viz.github.io/ <br>(branca) https://pypi.org/project/branca/ <br>(gpxpy) https://github.com/tkrajina/gpxpy <br>(vincenty) https://pypi.org/project/vincenty/ <br>(folium) https://github.com/python-visualization/folium <br>(humanfriendly) https://pypi.org/project/humanfriendly/ <br>(reverse-geocode) https://pypi.org/project/reverse-geocode/ <br>