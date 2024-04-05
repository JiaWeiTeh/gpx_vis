About `gpx_vis`===============`gpx_vis` allows simple and easy visualisation of your cycling/walking tours.Important: currently only works for .GPX from completed paths in Komoot.Install-------```bashpip install gpx_vis```Example-------```pythonfrom gpx_vis import Tracktrack = Track(pathname)````pathname` serves as a reference to either a single `.gpx` file or a directory.If a directory path is provided, all .gpx files contained within that directorywill be merged, but retain their original metadata (e.g., different tracks remain separated).This allows users to overlay multiple tracks onto a single map, providing a comprehensive summary of the data being analyzed or created.