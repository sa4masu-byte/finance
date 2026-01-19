"""
WSGI configuration for PythonAnywhere deployment
"""
import sys
from pathlib import Path

# Add your project directory to the sys.path
project_home = '/home/YOUR_USERNAME/finance'  # ← あなたのユーザー名に変更
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Import Flask app
from dashboard.app import app as application  # noqa
