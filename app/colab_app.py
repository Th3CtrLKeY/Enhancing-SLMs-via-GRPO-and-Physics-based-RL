"""
Lightweight Colab-oriented entrypoint.

In Google Colab (T4 runtime), after cloning/uploading the repo:

    %cd /content/MTP   # or your path
    !pip install -q -r requirements-app.txt
    import os
    os.environ["GRADIO_SHARE"] = "1"
    !python -m app.colab_app

This reuses the same Gradio UI with a temporary public share link.
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_SHARE", "1")

from app.gradio_app import main

if __name__ == "__main__":
    main()
