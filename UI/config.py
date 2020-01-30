"""App configuration."""
from os import environ
import pylibmc


class Config:
    """Set Flask configuration vars from .env file."""

    # General Config
    SECRET_KEY = environ.get("SECRET_KEY")
    FLASK_APP = environ.get("FLASK_APP")
    FLASK_ENV = environ.get("FLASK_ENV")

    # Flask-Session
    mc = pylibmc.Client(
        ["127.0.0.1"], binary=True, behaviors={"tcp_nodelay": True, "ketama": True}
    )
    SESSION_TYPE = "memcached"
    SESSION_MEMCACHED = mc
