"""Streamlit Community Cloud entrypoint.

The production UI lives in src/ui.py. Importing it runs the Streamlit app
without duplicating dashboard logic.
"""

import src.ui  # noqa: F401
