"""gpx_vis: turn ``.gpx`` tracks into an interactive HTML map.

>>> from gpx_vis import Track
>>> Track("my_ride.gpx").create_map("my_ride.html")
"""
from .track import Track

__all__ = ["Track"]
__version__ = "0.4.0"
