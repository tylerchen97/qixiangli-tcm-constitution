import pytest

def test_app_builds():
    pytest.importorskip("gradio")
    import app
    assert app.demo is not None
