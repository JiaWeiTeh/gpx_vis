# gpx_vis

Turn your `.gpx` tracks into an interactive HTML map, coloured by speed, with a speed and elevation chart for every route.

Works with exports from [Komoot](https://www.komoot.com/), [Strava](https://www.strava.com/), [gpx.studio](https://gpx.studio/) or any GPS device, as long as the file has positions and timestamps. Elevation is optional.

![An example ride replayed: the route coloured by speed, with speed and elevation over time](https://raw.githubusercontent.com/JiaWeiTeh/gpx_vis/main/media/demo.gif)

[**Open the live example map**](https://raw.githack.com/JiaWeiTeh/gpx_vis/main/example/map.html) (five rides around Boston)

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

**Interactive map.** Each route is coloured by speed (purple = slow, yellow = fast), on one scale for the whole map shown in the legend. Hover a route to see its name. Click the flag at its end for a summary and a chart of speed and elevation over time. Hover the chart to read exact values:

<img src="https://raw.githubusercontent.com/JiaWeiTeh/gpx_vis/main/media/popup.png" alt="Route popup: summary, speed and elevation over time, hover readout" width="420">

Prefer elevation colours on the map?

```python
track.create_map("my_ride.html", color_by="elevation")
```

**Backgrounds.** Switch in the bottom-right corner between OpenStreetMap (default), Plain, Dark mode, Plain (hierarchical) and Topographic.

| Dark mode | OpenStreetMap | Plain | Plain (hierarchical) |
|---|---|---|---|
| ![](https://raw.githubusercontent.com/JiaWeiTeh/gpx_vis/main/media/ex_dark.png) | ![](https://raw.githubusercontent.com/JiaWeiTeh/gpx_vis/main/media/ex_openstreet.png) | ![](https://raw.githubusercontent.com/JiaWeiTeh/gpx_vis/main/media/ex_plain.png) | ![](https://raw.githubusercontent.com/JiaWeiTeh/gpx_vis/main/media/ex_plainheir.png) |

**Lite mode** for very large files. Each route is drawn with at most about `nlite` points (default 50, minimum 10):

```python
track.create_map("my_ride.html", lite=True, nlite=100)
```

**Route summary** as a table: distance, total and moving time, average and top speed.

```python
track.summary
```

**Cities you passed through**, sorted by country then name, with the time spent in each:

```python
track.city_list
```

**Raw data** for your own analysis: `track.t` (time, UTC), `track.y` (latitude, deg), `track.x` (longitude, deg), `track.z` (elevation, m), `track.speed` (km/h), or everything except speed as a `pandas.DataFrame` in `track.data`.

## Try the example

```bash
python example/example.py                        # example/data -> example/map.html
python example/example.py path/to/tracks out.html

pip install matplotlib
python example/demo_animation.py                 # the animation above -> media/demo.gif
python example/demo_animation.py path/to/tracks "Morning Ride" ride.gif
```

## Put a map on your website

The map is a single HTML file. Put it in a GitHub Pages repository and embed it in any page:

```html
<iframe src="https://YOUR-NAME.github.io/rides/map.html" width="100%" height="500" style="border:0"></iframe>
```

For a lighter preview, use a GIF made with `example/demo_animation.py`.

## How the numbers are computed

- **Distance** adds up the geodesic steps between GPS points. Gaps between track segments, such as a paused recording, are not bridged or counted.
- **Stops.** Staying within 15 m of one spot for at least a minute counts as a stop. No distance is added and the time does not count as moving time. Without this, GPS drift while standing still adds hundreds of metres.
- **Speed** at each point is the distance covered in the 30 seconds around it, divided by that time. Raw point-to-point GPS speed is too noisy to use.
- **Limit.** GPS jitter while walking slowly can still add a few percent to the distance.

## Notes

- **Your files need timestamps.** Exports of recorded activities have them. Planned routes usually do not. All times are shown in UTC.
- **Progress messages** go through Python's `logging`. To see them, add `import logging; logging.basicConfig(level=logging.INFO)` before creating the map.
- **CARTO backgrounds** (Plain, Dark mode, Plain (hierarchical)) officially require a free [CARTO API key](https://carto.com/basemaps/apikey/). Pass it with `track.create_map("my_ride.html", carto_api_key="...")` or set the `CARTO_API_KEY` environment variable. OpenStreetMap and Topographic need no key. The key is written into the HTML file.
- **Privacy.** The HTML file and the animation show every point of your tracks, and the HTML file contains their timestamps. Your browser fetches map tiles from OpenStreetMap, CARTO and OpenTopoMap while viewing it. Think twice before sharing a map that starts or ends at your home.

## License

[BSD 3-Clause](LICENSE).
