import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))

from fastapi.testclient import TestClient
from server import app


def test_admin_page_returns_html():
    client = TestClient(app)
    response = client.get("/admin")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_admin_post_updates_config():
    client = TestClient(app)
    response = client.post(
        "/admin",
        data={
            "provider": "deepseek",
            "deepseek_api_key": "demo-key-test",
            "deepseek_model": "deepseek-v4-flash",
            "whisper_model": "small",
            "vad_threshold": "0.6",
        },
        follow_redirects=False,
    )
    assert response.status_code in (200, 302, 303)


def test_subtitle_page_returns_html():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "subtitle" in response.text.lower()


def test_websocket_route_exists():
    routes = [r.path for r in app.routes]
    assert "/ws" in routes


def test_admin_route_exists():
    routes = [r.path for r in app.routes]
    assert "/admin" in routes


def test_root_route_exists():
    routes = [r.path for r in app.routes]
    assert "/" in routes
