"""Build an interactive map from the sample tracks in ``example/data``.

Usage (from any directory):

    python example/example.py                       # example/data -> example/map.html
    python example/example.py path/to/tracks        # a folder of .gpx files, or one file
    python example/example.py path/to/tracks out.html
"""
import logging
import sys
from pathlib import Path

from gpx_vis import Track

HERE = Path(__file__).resolve().parent


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    data = Path(argv[0]) if len(argv) > 0 else HERE / "data"
    output = Path(argv[1]) if len(argv) > 1 else HERE / "map.html"

    # show progress messages ("Reading data...", "File saved as ...")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    track = Track(data)
    track.create_map(output, lite=True, nlite=200)
    # distance, total and moving time, average and top speed per route
    columns = ['name', 'distance_km', 'moving_time', 'moving_speed_kmh', 'max_speed_kmh']
    print(track.summary[columns].round(1).to_string(index=False))
    # cities passed through, with the time spent in each:
    # print(track.city_list)


if __name__ == "__main__":
    main()
