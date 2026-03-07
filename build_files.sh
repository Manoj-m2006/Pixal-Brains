#!/bin/bash
# Vercel build script - runs before deployment
set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Collecting static files..."
cd pixel_brains_django
python manage.py collectstatic --noinput

echo "Build complete."
