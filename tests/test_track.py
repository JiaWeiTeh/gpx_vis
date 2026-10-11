"""Tests for the gpx_vis Track class (run with ``pytest`` from the repo root)."""
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import gpxpy
import gpxpy.gpx
import numpy as np
import pytest
from vincenty import vincenty

from gpx_vis import Track


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'example', 'data')
SINGLE_FILE = os.path.join(DATA_DIR, 'parkwalk.gpx')


class TestTrackInit:
    """Tests for Track initialisation and input validation."""

    def test_load_single_file(self):
        track = Track(SINGLE_FILE)
        assert len(track.x) > 0
        assert len(track.y) == len(track.x)
        assert len(track.z) == len(track.x)
        assert len(track.t) == len(track.x)
        assert len(track.name) == len(track.x)

    def test_load_directory(self):
        track = Track(DATA_DIR)
        assert len(track.x) > 0

    def test_load_pathlib_path(self):
        track = Track(Path(SINGLE_FILE))
        assert len(track.x) > 0

    def test_invalid_path_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            Track('/nonexistent/path')

    def test_non_gpx_file_raises_value_error(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a gpx file")
        with pytest.raises(ValueError):
            Track(str(txt_file))

    def test_empty_directory_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError):
            Track(str(tmp_path))

    def test_non_string_pathname_raises_type_error(self):
        with pytest.raises(TypeError):
            Track(123)

    def test_none_pathname_raises_type_error(self):
        with pytest.raises(TypeError):
            Track(None)


class TestTrackData:
    """Tests for Track data properties."""

    @pytest.fixture
    def track(self):
        return Track(SINGLE_FILE)

    def test_data_returns_dataframe(self, track):
        df = track.data
        assert list(df.columns) == ['trackName', 'latitude', 'longitude', 'elevation', 'time']

    def test_data_is_cached(self, track):
        df1 = track.data
        df2 = track.data
        assert df1 is df2

    def test_header_returns_column_names(self, track):
        headers = track.header
        assert 'latitude' in headers
        assert 'longitude' in headers

    def test_arrays_are_numpy(self, track):
        assert isinstance(track.x, np.ndarray)
        assert isinstance(track.y, np.ndarray)
        assert isinstance(track.z, np.ndarray)


class TestTrackSplit:
    """Tests for track splitting logic."""

    @pytest.fixture
    def single_track(self):
        return Track(SINGLE_FILE)

    @pytest.fixture
    def multi_track(self):
        return Track(DATA_DIR)

    def test_single_track_split(self, single_track):
        splits = single_track.idx_trksplit()
        assert len(splits) >= 1
        # first segment starts at 0
        assert splits[0][0] == 0
        # last segment ends at total length
        assert splits[-1][1] == len(single_track.name)

    def test_multi_track_no_gaps(self, multi_track):
        splits = multi_track.idx_trksplit()
        # segments must cover all points with no gaps or overlaps
        for i in range(len(splits) - 1):
            assert splits[i][1] == splits[i + 1][0]
        assert splits[0][0] == 0
        assert splits[-1][1] == len(multi_track.name)

    def test_multi_track_one_segment_per_file(self, multi_track):
        splits = multi_track.idx_trksplit()
        assert len(splits) == len([f for f in os.listdir(DATA_DIR) if f.endswith('.gpx')])


class TestCreateMap:
    """Tests for map creation."""

    @pytest.fixture
    def track(self):
        return Track(SINGLE_FILE)

    def test_create_map_generates_file(self, track, tmp_path):
        output = str(tmp_path / "test_map.html")
        track.create_map(output)
        assert os.path.exists(output)
        assert os.path.getsize(output) > 0

    def test_create_map_returns_output_path(self, track, tmp_path):
        output = str(tmp_path / "test_map.html")
        assert track.create_map(output) == output

    def test_create_map_adds_html_extension(self, track, tmp_path):
        output = str(tmp_path / "test_map")
        result = track.create_map(output)
        assert result == output + ".html"
        assert os.path.exists(output + ".html")

    def test_create_map_accepts_pathlib_path(self, track, tmp_path):
        output = tmp_path / "test_map.html"
        track.create_map(output)
        assert output.exists()

    def test_create_map_lite_mode(self, track, tmp_path):
        output = str(tmp_path / "test_map_lite.html")
        track.create_map(output, lite=True, nlite=20)
        assert os.path.exists(output)

    def test_create_map_lite_mode_writes_fewer_points(self, track, tmp_path):
        full = (tmp_path / "full.html")
        lite = (tmp_path / "lite.html")
        track.create_map(full)
        track.create_map(lite, lite=True, nlite=20)
        assert lite.stat().st_size < full.stat().st_size

    def test_create_map_invalid_lite_type(self, track, tmp_path):
        output = str(tmp_path / "test_map.html")
        with pytest.raises(TypeError):
            track.create_map(output, lite="yes")

    def test_create_map_nlite_too_small(self, track, tmp_path):
        output = str(tmp_path / "test_map.html")
        with pytest.raises(ValueError):
            track.create_map(output, lite=True, nlite=5)

    def test_create_map_nlite_not_integer(self, track, tmp_path):
        output = str(tmp_path / "test_map.html")
        with pytest.raises(TypeError):
            track.create_map(output, lite=True, nlite=20.5)

    def test_create_map_background_layers(self, track, tmp_path):
        output = tmp_path / "layers.html"
        track.create_map(output)
        html = output.read_text()
        for name in ("Plain", "Dark mode", "Plain (hierarchical)", "Topographic"):
            assert name in html
        assert "key=" not in html  # no API key written unless one was given

    def test_create_map_carto_api_key_in_tile_urls(self, track, tmp_path, monkeypatch):
        monkeypatch.delenv("CARTO_API_KEY", raising=False)
        output = tmp_path / "keyed.html"
        track.create_map(output, carto_api_key="TESTKEY123")
        assert "key=TESTKEY123" in output.read_text()


class TestCityList:
    """Tests for reverse-geocoded city summary."""

    def test_city_list_sorted_with_frequencies(self):
        cities = Track(SINGLE_FILE).city_list
        assert len(cities) > 0
        assert all(c.frequency > 0 for c in cities)
        assert sum(c.frequency for c in cities) == len(Track(SINGLE_FILE).x)
        assert cities == sorted(cities)


# =============================================================================
# Speed (physics) tests on synthetic tracks with known answers
# =============================================================================

def _write_gpx(path, tracks):
    """Write a GPX file. tracks: list of (name, [(lat, lon, elevation, datetime), ...])."""
    gpx = gpxpy.gpx.GPX()
    for name, points in tracks:
        trk = gpxpy.gpx.GPXTrack(name=name)
        seg = gpxpy.gpx.GPXTrackSegment()
        trk.segments.append(seg)
        gpx.tracks.append(trk)
        for lat, lon, ele, t in points:
            seg.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon, elevation=ele, time=t))
    path.write_text(gpx.to_xml())
    return str(path)


T0 = datetime(2024, 6, 1, 8, 0, 0, tzinfo=timezone.utc)
STEP_DEG = 0.0005  # along a meridian at 49 deg N: about 55.6 m per step


def _straight_line(n, step_s, lat0=49.0, lon=8.0, t0=T0):
    """n points going due north, one step every step_s seconds."""
    return [(lat0 + i * STEP_DEG, lon, 100.0, t0 + timedelta(seconds=i * step_s)) for i in range(n)]


def _expected_kmh(step_s, lat0=49.0, lon=8.0):
    return vincenty((lat0, lon), (lat0 + STEP_DEG, lon)) / step_s * 3600


class TestSpeed:
    def test_constant_speed_is_recovered(self, tmp_path):
        track = Track(_write_gpx(tmp_path / "c.gpx", [("c", _straight_line(200, 5))]))
        expected = _expected_kmh(5)  # about 40 km/h
        assert 39 < expected < 41
        np.testing.assert_allclose(track.speed, expected, rtol=1e-3)

    def test_speed_change_is_tracked(self, tmp_path):
        slow = _straight_line(100, 20)               # about 10 km/h
        t_switch = slow[-1][3]
        lat_switch = slow[-1][0]
        fast = [(lat_switch + (i + 1) * STEP_DEG, 8.0, 100.0, t_switch + timedelta(seconds=(i + 1) * 5))
                for i in range(100)]                  # about 40 km/h
        track = Track(_write_gpx(tmp_path / "s.gpx", [("s", slow + fast)]))
        # away from the switch (further than half the smoothing window) speeds are exact
        np.testing.assert_allclose(track.speed[:90], _expected_kmh(20), rtol=1e-3)
        np.testing.assert_allclose(track.speed[110:], _expected_kmh(5), rtol=1e-3)

    def test_stationary_is_zero(self, tmp_path):
        pts = [(49.0, 8.0, 100.0, T0 + timedelta(seconds=i)) for i in range(50)]
        track = Track(_write_gpx(tmp_path / "still.gpx", [("still", pts)]))
        np.testing.assert_allclose(track.speed, 0.0, atol=1e-9)

    def test_speed_does_not_leak_between_tracks(self, tmp_path):
        a = ("a", _straight_line(60, 20))                                   # about 10 km/h
        b = ("b", _straight_line(60, 5, lat0=52.0, t0=T0 + timedelta(days=1)))  # about 40 km/h, far away
        track = Track(_write_gpx(tmp_path / "two.gpx", [a, b]))
        np.testing.assert_allclose(track.speed[:60], _expected_kmh(20), rtol=1e-3)
        np.testing.assert_allclose(track.speed[60:], _expected_kmh(5, lat0=52.0), rtol=1e-3)

    def test_duplicate_timestamps_give_finite_speed(self, tmp_path):
        pts = _straight_line(50, 5)
        pts[20] = (pts[20][0], pts[20][1], pts[20][2], pts[19][3])  # same time as previous point
        pts[21] = (pts[21][0], pts[21][1], pts[21][2], pts[19][3])
        track = Track(_write_gpx(tmp_path / "dup.gpx", [("dup", pts)]))
        assert np.isfinite(track.speed).all()
        assert (track.speed >= 0).all()

    def test_missing_timestamps_give_nan_and_map_raises(self, tmp_path):
        pts = [(lat, lon, ele, None) for lat, lon, ele, _ in _straight_line(20, 5)]
        track = Track(_write_gpx(tmp_path / "notime.gpx", [("notime", pts)]))
        assert np.isnan(track.speed).all()
        with pytest.raises(ValueError, match="timestamp"):
            track.create_map(tmp_path / "notime.html")

    def test_mean_speed_matches_distance_over_time(self):
        # on real data the time-weighted mean of the smoothed speed should agree
        # with total distance / total time (the popup's "Average speed")
        track = Track(os.path.join(DATA_DIR, 'cyclingBoston.gpx'))
        seconds = np.array([(t - track.t[0]).total_seconds() for t in track.t])
        weighted_mean = np.sum(track.speed[1:] * np.diff(seconds)) / seconds[-1]
        overall = track._step_km.sum() / (seconds[-1] / 3600)
        assert abs(weighted_mean - overall) / overall < 0.02

    def test_example_walk_is_walking_pace(self):
        # parkwalk.gpx was timed by gpx.studio at a constant 5 km/h
        speed = Track(SINGLE_FILE).speed
        assert 4.9 < np.median(speed) < 5.1


class TestColorBy:
    @pytest.fixture
    def track(self):
        return Track(SINGLE_FILE)

    def test_speed_is_default_with_legend_and_utc_axis(self, track, tmp_path):
        html = (tmp_path / "m.html")
        track.create_map(html)
        text = html.read_text()
        assert "Speed (km/h)" in text      # legend caption and chart axis
        assert "Average speed" in text     # popup subtitle
        assert "Time (UTC)" in text

    def test_elevation_option(self, track, tmp_path):
        html = (tmp_path / "e.html")
        track.create_map(html, color_by="elevation")
        text = html.read_text()
        assert "Elevation (m)" in text
        assert "Speed (km/h)" not in text

    def test_invalid_color_by(self, track, tmp_path):
        with pytest.raises(ValueError):
            track.create_map(tmp_path / "x.html", color_by="heart rate")
