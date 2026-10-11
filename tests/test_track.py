"""Tests for the gpx_vis Track class (run with ``pytest`` from the repo root)."""
import os
from pathlib import Path

import numpy as np
import pytest

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
