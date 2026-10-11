"""
Open a generated map in headless Chromium and check that it actually works:
no JavaScript errors, routes and legend drawn, and a popup chart that renders
and responds to the mouse. Needs Playwright and network access to the CDNs:

    pip install playwright && playwright install chromium
    pytest tests/test_browser.py

Skipped when Playwright or its browser is not installed.
"""
import os
import re

import pytest

from gpx_vis import Track

sync_api = pytest.importorskip("playwright.sync_api")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'example', 'data')

OPEN_POPUP_JS = """(name) => {
    const cluster = window[Object.keys(window).find(k => k.startsWith('marker_cluster_'))];
    for (const m of cluster.getLayers()) {
        if (!(m.getPopup && m.getPopup())) continue;
        const el = m.getPopup().getContent();
        const html = typeof el === 'string' ? el : el.outerHTML;
        const src = (html.match(/base64,([A-Za-z0-9+\\/=]+)/) || [])[1];
        if (src && atob(src).includes(name)) {
            cluster.zoomToShowLayer(m, () => m.openPopup());
            return true;
        }
    }
    return false;
}"""


def check_map(page, html_path, route_name='cyclingBoston'):
    """Assertions shared by the test and by offline runs that serve the CDN files locally."""
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto('file://' + os.path.abspath(html_path))
    page.wait_for_function("() => typeof L !== 'undefined' && document.querySelector('.legend')", timeout=30000)
    page.wait_for_timeout(500)
    assert not errors, errors
    assert page.locator('path.leaflet-interactive').count() > 0
    assert 'Speed (km/h)' in page.locator('.legend').inner_text()

    assert page.evaluate(OPEN_POPUP_JS, route_name)
    frame = None
    for _ in range(100):
        frame = next((f for f in page.frames if f.url.startswith('data:text/html')), None)
        if frame:
            break
        page.wait_for_timeout(100)
    assert frame is not None, "popup iframe did not open"
    chart = frame.wait_for_selector('svg.marks, canvas', timeout=30000)
    box = chart.bounding_box()
    page.mouse.move(box['x'] + box['width'] * 0.6, box['y'] + box['height'] * 0.4)
    frame.wait_for_selector('#vg-tooltip-element.visible', timeout=10000)
    tooltip = frame.inner_text('#vg-tooltip-element')
    for label in ('Speed (km/h)', 'Elevation (m)', 'Time (UTC)'):
        assert label in tooltip
    assert not errors, errors


def test_map_and_popup_chart_render(tmp_path):
    html = Track(DATA_DIR).create_map(tmp_path / 'map.html')
    with sync_api.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(executable_path=os.environ.get('CHROMIUM_PATH') or None)
        except Exception as exc:  # browser not installed
            if os.environ.get('GPX_VIS_REQUIRE_BROWSER'):
                raise
            pytest.skip(f"Chromium not available: {exc}")
        page = browser.new_page(viewport={'width': 1200, 'height': 800})
        # map tiles are not needed for these checks
        page.route(re.compile(r'(tile|basemaps|cartocdn|opentopomap)'), lambda route: route.abort())
        try:
            check_map(page, html)
        finally:
            browser.close()
