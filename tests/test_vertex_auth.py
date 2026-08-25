"""Tests for Vertex service-account token helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent_metering import vertex_auth


def test_get_access_token_uses_cache(tmp_path):
    vertex_auth.clear_token_cache()
    sa = tmp_path / "sa.json"
    sa.write_text(
        '{"type":"service_account","client_email":"a@b.c","private_key_id":"1"}',
        encoding="utf-8",
    )

    fake_creds = MagicMock()
    fake_creds.token = "tok-1"
    fake_creds.expiry = None

    with patch(
        "google.oauth2.service_account.Credentials.from_service_account_info"
    ) as from_info:
        from_info.return_value = fake_creds
        with patch("google.auth.transport.requests.Request"):
            t1 = vertex_auth.get_access_token(sa)
            t2 = vertex_auth.get_access_token(sa)
    assert t1 == "tok-1"
    assert t2 == "tok-1"
    assert fake_creds.refresh.call_count == 1


def test_get_access_token_force_refresh():
    vertex_auth.clear_token_cache()
    info = {"type": "service_account", "client_email": "a@b.c", "private_key_id": "2"}

    fake_creds = MagicMock()
    fake_creds.token = "tok-2"
    fake_creds.expiry = None

    with patch(
        "google.oauth2.service_account.Credentials.from_service_account_info"
    ) as from_info:
        from_info.return_value = fake_creds
        with patch("google.auth.transport.requests.Request"):
            vertex_auth.get_access_token(info)
            vertex_auth.get_access_token(info, force_refresh=True)
    assert fake_creds.refresh.call_count == 2


def test_get_access_token_missing_google_auth():
    vertex_auth.clear_token_cache()
    with patch.dict(
        "sys.modules",
        {
            "google": None,
            "google.auth": None,
            "google.auth.transport": None,
            "google.auth.transport.requests": None,
            "google.oauth2": None,
            "google.oauth2.service_account": None,
        },
    ):
        with pytest.raises(ImportError, match="google-auth"):
            vertex_auth.get_access_token(
                {"type": "service_account", "client_email": "x", "private_key_id": "y"}
            )
