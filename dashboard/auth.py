"""Simple multi-user authentication gate for Streamlit sessions."""

import hmac
import os
from pathlib import Path

import streamlit as st


def _load_project_env_file() -> None:
    """Load simple KEY=VALUE entries from project .env into os.environ."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _read_secret(key: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then environment variables."""
    try:
        return str(st.secrets[key])
    except Exception:
        # Covers missing secrets file and missing keys across Streamlit versions.
        pass
    return os.getenv(key, default)


def _default_users() -> dict[str, str]:
    """Default local users for quick team access."""
    return {
        "hiep": "12345",
        "binh": "12345",
        "chuong": "12345",
        "bella": "12345",
    }


def _read_users_from_env() -> dict[str, str]:
    """
    Read multi-user credentials from APP_LOGIN_CREDENTIALS.

    Format: user1:pass1,user2:pass2,user3:pass3
    """
    raw = _read_secret("APP_LOGIN_CREDENTIALS", "").strip()
    if not raw:
        return {}

    users: dict[str, str] = {}
    for pair in raw.split(","):
        item = pair.strip()
        if not item or ":" not in item:
            continue
        username, password = item.split(":", 1)
        username = username.strip()
        password = password.strip()
        if username and password:
            users[username] = password
    return users


def require_login() -> bool:
    """
    Render login form and guard access.

    Returns ``True`` only when the current session is authenticated.
    If auth config is missing, access is denied by default.
    """
    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False
    if "auth_username" not in st.session_state:
        st.session_state.auth_username = ""

    _load_project_env_file()

    # Multi-user mode only: APP_LOGIN_CREDENTIALS or default team users.
    multi_users = _read_users_from_env()
    if not multi_users:
        multi_users = _default_users()

    if st.session_state.is_authenticated:
        with st.sidebar:
            st.success(f"Signed in as {st.session_state.auth_username}")
            if st.button("Sign out", use_container_width=True):
                st.session_state.is_authenticated = False
                st.session_state.auth_username = ""
                st.rerun()
        return True

    st.title("Sign in")
    st.caption("Please sign in to continue.")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign in", type="primary")

    if submit:
        username_input = username.strip()
        password_input = password.strip()

        login_ok = False
        if username_input in multi_users:
            login_ok = hmac.compare_digest(password_input, multi_users[username_input])

        if login_ok:
            st.session_state.is_authenticated = True
            st.session_state.auth_username = username_input
            st.success("Sign in successful.")
            st.rerun()
        st.error("Invalid username or password.")

    return False
