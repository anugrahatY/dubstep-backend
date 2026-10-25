#!/usr/bin/env python3
"""ASGI entry point. Run from this directory: uvicorn server:app --reload"""

from api import create_app

app = create_app()
