"""
Vercel serverless entry point for the Django application.
This file is the WSGI bridge between Vercel and Django.
"""
import os
import sys

# Add repo root and Django project to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DJANGO_DIR = os.path.join(ROOT_DIR, "pixel_brains_django")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, DJANGO_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pixel_brains_django.settings")

from django.core.wsgi import get_wsgi_application

app = get_wsgi_application()
