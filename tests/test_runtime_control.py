import json

import pytest

from xau_company import runtime_control


class _Response:
    def __init__(self, text: str, payload=None):
        self.text = text
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        if self._payload is not None:
            return self._payload
        return json.loads(self.text)


class _Session:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_parse_enabled_requires_boolean():
    assert runtime_control.parse_enabled('{"enabled": true}') is True
    assert runtime_control.parse_enabled({"enabled": False}) is False
    with pytest.raises(ValueError):
        runtime_control.parse_enabled('{"enabled": "yes"}')


def test_fetch_enabled_cache_busts_and_disables_cache():
    session = _Session(_Response('{"enabled": false}'))
    assert runtime_control.fetch_enabled("https://example.test/control.json", session=session) is False
    url, kwargs = session.calls[0]
    assert "_ts=" in url
    assert kwargs["headers"]["Cache-Control"] == "no-cache"


def test_runtime_config_rejects_dangerously_fast_poll():
    cfg = runtime_control.RuntimeControlConfig(poll_seconds=1)
    with pytest.raises(ValueError):
        cfg.validate()
