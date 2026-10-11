"""Tests for the gpx_vis Track class (run with ``pytest`` from the repo root)."""
import base64
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import gpxpy
import gpxpy.gpx
import altair as alt
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
    """
    Write a GPX file. tracks: list of (name, points), where points is a list of
    (lat, lon, elevation, datetime) for one segment, or a list of such lists for
    several segments.
    """
    gpx = gpxpy.gpx.GPX()
    for name, points in tracks:
        trk = gpxpy.gpx.GPXTrack(name=name)
        gpx.tracks.append(trk)
        segments = points if points and isinstance(points[0], list) else [points]
        for seg_points in segments:
            seg = gpxpy.gpx.GPXTrackSegment()
            trk.segments.append(seg)
            for lat, lon, ele, t in seg_points:
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


def _popup_pages(html_text):
    """The HTML pages embedded (base64) in the map's popup iframes."""
    return [base64.b64decode(b).decode() for b in
            re.findall(r'data:text/html;charset=utf-8;base64,([A-Za-z0-9+/=]+)', html_text)]


class TestColorBy:
    @pytest.fixture
    def track(self):
        return Track(SINGLE_FILE)

    def test_speed_is_default_with_legend(self, track, tmp_path):
        text = Path(track.create_map(tmp_path / "m.html")).read_text()
        assert "Speed (km/h)" in text      # legend caption

    def test_elevation_option(self, track, tmp_path):
        text = Path(track.create_map(tmp_path / "e.html", color_by="elevation")).read_text()
        assert "Elevation (m)" in text
        assert "Speed (km/h)" not in text  # legend switched (popup pages are base64)

    def test_invalid_color_by(self, track, tmp_path):
        with pytest.raises(ValueError):
            track.create_map(tmp_path / "x.html", color_by="heart rate")


class TestPopupChart:
    @pytest.fixture
    def pages(self, tmp_path):
        return _popup_pages(Path(Track(DATA_DIR).create_map(tmp_path / "m.html")).read_text())

    def test_one_popup_per_route(self, pages):
        assert len(pages) == 5

    def test_speed_and_elevation_panels_in_utc(self, pages):
        page = pages[0]
        for text in ("Speed (km/h)", "Elevation (m)", "Time (UTC)", "Moving average", '"utc"'):
            assert text in page

    def test_loads_vega_versions_matching_altair(self, pages):
        # folium.VegaLite loaded Vega-Lite 2 for Altair 6 charts, which then failed to draw
        major = alt.VEGALITE_VERSION.split('.')[0]
        assert f"vega-lite@{major}" in pages[0]
        assert f"vega-lite/v{major}" in pages[0]  # the spec's schema

    def test_geojson_features_have_properties(self, tmp_path):
        # folium's JS reads feature.properties.style; a bare geometry crashed the whole map
        text = Path(Track(DATA_DIR).create_map(tmp_path / "m.html")).read_text()
        payloads = re.findall(r'_add\((\{"features".*?\})\);', text)
        assert payloads
        for payload in payloads:
            for feature in json.loads(payload)["features"]:
                assert isinstance(feature.get("properties"), dict)


# =============================================================================
# Regression tests for the route, distance and timing fixes
# =============================================================================

def _ar1_noise(n, sigma_m, tau_s, seed):
    """GPS-like position error in metres: correlated in time (AR(1) process)."""
    rng = np.random.default_rng(seed)
    rho = np.exp(-1 / tau_s)
    e = np.zeros((n, 2))
    e[0] = rng.normal(0, sigma_m, 2)
    for i in range(1, n):
        e[i] = rho * e[i - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0, sigma_m, 2)
    return e


def _from_metres(east, north, lat0=49.0, lon0=8.0):
    m_per_deg = 111195.0
    return lat0 + north / m_per_deg, lon0 + east / (m_per_deg * np.cos(np.radians(lat0)))


class TestSameNameTracks:
    def test_files_with_the_same_track_name_stay_separate(self, tmp_path):
        folder = tmp_path / "rides"
        folder.mkdir()
        _write_gpx(folder / "a.gpx", [("Morning Ride", _straight_line(50, 10))])
        _write_gpx(folder / "b.gpx", [("Morning Ride", _straight_line(50, 10, lat0=52.0, t0=T0 + timedelta(days=3)))])
        track = Track(folder)
        assert len(track.idx_trksplit()) == 2
        one_ride = vincenty((49.0, 8.0), (49.0 + 49 * STEP_DEG, 8.0))
        np.testing.assert_allclose(track.summary["distance_km"], one_ride, rtol=1e-3)


class TestSegments:
    def test_gap_between_segments_is_not_counted_or_drawn(self, tmp_path):
        first = _straight_line(30, 5)
        second = _straight_line(30, 5, lat0=49.1, t0=T0 + timedelta(hours=1))  # 11 km away, an hour later
        track = Track(_write_gpx(tmp_path / "gap.gpx", [("gap", [first, second])]))
        assert len(track.idx_trksplit()) == 1
        assert len(track._segments()) == 2
        per_segment = vincenty((49.0, 8.0), (49.0 + 29 * STEP_DEG, 8.0))
        assert track.summary["distance_km"][0] == pytest.approx(2 * per_segment, rel=1e-3)
        # the hour-long gap is not moving time
        assert track.summary["moving_time"][0] == timedelta(seconds=2 * 29 * 5)
        np.testing.assert_allclose(track.speed, _expected_kmh(5), rtol=1e-3)


class TestMissingElevation:
    def test_missing_elevation_is_nan_not_zero(self, tmp_path):
        pts = _straight_line(40, 5)
        pts[10] = (pts[10][0], pts[10][1], None, pts[10][3])
        track = Track(_write_gpx(tmp_path / "noele.gpx", [("noele", pts)]))
        assert np.isnan(track.z[10])
        assert np.nanmin(track.z) == 100.0
        html = track.create_map(tmp_path / "e.html", color_by="elevation")
        assert os.path.getsize(html) > 0


class TestStops:
    def test_standing_still_with_gps_drift_adds_nothing(self, tmp_path):
        noise = _ar1_noise(600, 3.0, 60, seed=1)  # 10 minutes, 3 m drift
        lat, lon = _from_metres(noise[:, 0], noise[:, 1])
        pts = [(a, b, 100.0, T0 + timedelta(seconds=i)) for i, (a, b) in enumerate(zip(lat, lon))]
        track = Track(_write_gpx(tmp_path / "still.gpx", [("still", pts)]))
        assert track.summary["distance_km"][0] < 0.02
        assert np.median(track.speed) < 0.5
        assert track.summary["moving_time"][0] < timedelta(minutes=1)

    def test_ride_stop_ride(self, tmp_path):
        # 10 min at 12 km/h, 10 min standing still, 10 min at 12 km/h, with 3 m drift
        n = 1800
        north = np.zeros(n)
        for i in range(1, n):
            north[i] = north[i - 1] + (0 if 600 <= i < 1200 else 12 / 3.6)
        noise = _ar1_noise(n, 3.0, 60, seed=2)
        lat, lon = _from_metres(noise[:, 0], north + noise[:, 1])
        pts = [(a, b, 100.0, T0 + timedelta(seconds=i)) for i, (a, b) in enumerate(zip(lat, lon))]
        stats = Track(_write_gpx(tmp_path / "rsr.gpx", [("rsr", pts)])).summary.iloc[0]
        true_km = north[-1] / 1000
        assert stats["distance_km"] == pytest.approx(true_km, rel=0.03)
        assert abs(stats["moving_time"] - timedelta(minutes=20)) < timedelta(minutes=1)
        assert stats["moving_speed_kmh"] == pytest.approx(12.0, rel=0.03)
        assert stats["avg_speed_kmh"] == pytest.approx(8.0, rel=0.03)

    def test_tight_switchbacks_keep_their_distance(self, tmp_path):
        # 20 m zigzags at walking pace: a minimum-movement filter would lose ~45 %
        t = np.arange(600.0)
        east = np.abs(((t * 5 / 3.6) % 40) - 20)
        north = t * 0.3
        lat, lon = _from_metres(east, north)
        pts = [(a, b, 100.0, T0 + timedelta(seconds=i)) for i, (a, b) in enumerate(zip(lat, lon))]
        track = Track(_write_gpx(tmp_path / "zig.gpx", [("zig", pts)]))
        true_km = np.sum(np.hypot(np.diff(east), np.diff(north))) / 1000
        assert track.summary["distance_km"][0] == pytest.approx(true_km, rel=0.01)


class TestCityTime:
    def test_city_time_adds_up_to_duration(self):
        track = Track(os.path.join(DATA_DIR, 'cyclingBoston.gpx'))
        cities = track.city_list
        total = sum((c.time for c in cities), timedelta(0))
        duration = track.summary["duration"][0]
        assert abs(total - duration) <= timedelta(seconds=len(cities))  # rounding to whole seconds

    def test_time_does_not_follow_point_count(self, tmp_path):
        # same time in two places, but one records 10x more points
        dense = [(49.0, 8.0 + i * 1e-6, 100.0, T0 + timedelta(seconds=i)) for i in range(600)]
        sparse = [(52.52, 13.40 + i * 1e-5, 100.0, T0 + timedelta(seconds=600 + 10 * (i + 1))) for i in range(60)]
        track = Track(_write_gpx(tmp_path / "two.gpx", [("two", dense + sparse)]))
        by_country = {c.country: c for c in track.city_list}
        assert by_country['Germany'].frequency > 1
        times = sorted(c.time for c in track.city_list)
        assert times[-1] - times[0] < timedelta(minutes=1)  # both about ten minutes


class TestLiteSampling:
    def test_budget_and_true_finish(self, tmp_path):
        track = Track(_write_gpx(tmp_path / "l.gpx", [("l", _straight_line(200, 1))]))
        idx = track._sample(0, 200, 50)
        assert len(idx) <= 50
        assert idx[0] == 0 and idx[-1] == 199

    def test_segment_ends_are_kept(self, tmp_path):
        segs = [_straight_line(100, 1), _straight_line(100, 1, lat0=49.1, t0=T0 + timedelta(hours=1))]
        track = Track(_write_gpx(tmp_path / "s.gpx", [("s", segs)]))
        idx = track._sample(0, 200, 20)
        for end in (0, 99, 100, 199):
            assert end in idx


class TestSummary:
    def test_summary_columns(self):
        summary = Track(DATA_DIR).summary
        assert list(summary.columns) == ['name', 'start', 'end', 'distance_km', 'duration', 'moving_time',
                                         'avg_speed_kmh', 'moving_speed_kmh', 'max_speed_kmh']
        assert len(summary) == 5
