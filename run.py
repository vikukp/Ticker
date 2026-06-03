"""Entry point — run the Streamlit app."""

import subprocess
import sys
import os

if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(__file__), "app", "ui.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
