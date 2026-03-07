"""
Core views for rendering Django pages.
Replicates the Streamlit UI functionality.
"""
from django.shortcuts import render
from django.http import HttpResponse
import json
from datetime import datetime

def index(request):
    """Main landing page."""
    return render(request, "index.html", {
        "active_tab": "home",
        "title": "Pixel Brains – Satellite Change Detection",
    })

def change_detection(request):
    """Tab 1: Change Detection - Draw on map, compare before/after."""
    return render(request, "change_detection.html", {
        "active_tab": "change_detection",
        "title": "Change Detection",
    })

def analysis_results(request):
    """Tab 2: Analysis Results - Show detailed Gemini AI analysis."""
    return render(request, "analysis_results.html", {
        "active_tab": "analysis_results",
        "title": "AI Analysis Results",
    })

def live_uplink(request):
    """Tab 3: Live Global Uplink - Real-time satellite imagery."""
    return render(request, "live_uplink.html", {
        "active_tab": "live_uplink",
        "title": "Live Global Uplink",
    })

def time_series_analysis(request):
    """Tab 4: Time Series Analysis - Visualize changes over multiple years."""
    current_year = datetime.now().year
    years = list(range(2017, current_year + 1))  # Sentinel-2 data available from 2017
    return render(request, "time_series_analysis.html", {
        "active_tab": "time_series",
        "title": "Time Series Analysis",
        "years": years,
    })

def disasters(request):
    """Tab 5: Natural Disasters monitoring."""
    return render(request, "disasters.html", {
        "active_tab": "disasters",
        "title": "Natural Disasters Monitoring",
    })
