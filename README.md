# gpx_vis

Turn your `.gpx` tracks into an interactive HTML map.

Works with exports from [Komoot](https://www.komoot.com/), [Strava](https://www.strava.com/), [gpx.studio](https://gpx.studio/) or any GPS device, as long as the file has latitude, longitude, elevation and time.

[**Live example**](https://raw.githack.com/JiaWeiTeh/gpx_vis/main/example/map.html) (five rides around Boston)

![Click the flag at the end of a route for details and an elevation profile](https://raw.githubusercontent.com/JiaWeiTeh/gpx_vis/main/media/track_info.png)

## Install

```bash
pip install gpx_vis
```

Requires Python 3.9 or newer. Dependencies (folium, gpxpy, pandas, ...) are installed automatically.

From source:

```bash
git clone https://github.com/JiaWeiTeh/gpx_vis.git
cd gpx_vis
pip install .
```

## Quick start

```python
from gpx_vis import Track

track = Track("my_ride.gpx")       # one .gpx file, or a folder of them
track.create_map("my_ride.html")   # open this file in your browser
```

Give `Track` a folder and every `.gpx` file inside is drawn on the same map, each as its own route.

## What you get

**Interactive map.** Each route is coloured by elevation (purple = low, yellow = high). Hover a route to see its name; click the flag at its end for start/end time and place, distance, duration and an elevation profile.

**Backgrounds.** Switch in the bottom-right corner between OpenStreetMap (default), Plain, Dark mode, Plain (hierarchical) and Topographic.

| Dark mode | OpenStreetMap | Plain | Plain (hierarchical) |
|---|---|---|---|
| ![](https://raw.githubusercontent.com/JiaWeiTeh/gpx_vis/main/media/ex_dark.png) | ![](https://raw.githubusercontent.com/JiaWeiTeh/gpx_vis/main/media/ex_openstreet.png) | ![](https://raw.githubusercontent.com/JiaWeiTeh/gpx_vis/main/media/ex_plain.png) | ![](https://raw.githubusercontent.com/JiaWeiTeh/gpx_vis/main/media/ex_plainheir.png) |

**Lite mode** for very large files. Each route is drawn with at most about `nlite` points (default 50, minimum 10):

```python
track.create_map("my_ride.html", lite=True, nlite=100)
```

**Cities you passed through**, sorted by country then name, with how many track points fell in each:

```python
track.city_list
```

**Raw data** for your own analysis: `track.t` (time, UTC), `track.y` (latitude, deg), `track.x` (longitude, deg), `track.z` (elevation, m), or everything at once as a `pandas.DataFrame` in `track.data`.

## Try the example

```bash
python example/example.py                        # example/data -> example/map.html
python example/example.py path/to/tracks out.html
```

## Notes

- **Progress messages** go through Python's `logging`. To see them, add `import logging; logging.basicConfig(level=logging.INFO)` before creating the map.
- **CARTO backgrounds** (Plain, Dark mode, Plain (hierarchical)) officially require a free [CARTO API key](https://carto.com/basemaps/apikey/). Pass it with `track.create_map("my_ride.html", carto_api_key="...")` or set the `CARTO_API_KEY` environment variable. OpenStreetMap and Topographic need no key. The key is written into the HTML file.
- **Privacy.** The HTML file contains every coordinate and timestamp of your tracks, and your browser fetches map tiles from OpenStreetMap, CARTO and OpenTopoMap while viewing it. Think twice before sharing a map that starts or ends at your home.

## License

[BSD 3-Clause](LICENSE).
