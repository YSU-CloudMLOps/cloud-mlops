"""Predictive maintenance API package."""

import sys

# Provide alias support for lowercase 'api' package import
if "API" in sys.modules and "api" not in sys.modules:
    sys.modules["api"] = sys.modules["API"]
