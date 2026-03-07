"""
Core app API URL configuration for AJAX endpoints.
"""
from django.urls import path
from . import api_views

urlpatterns = [
    path("reverse-geocode/", api_views.reverse_geocode, name="reverse_geocode"),
    path("fetch-images/", api_views.fetch_satellite_images, name="fetch_images"),
    path("run-detection/", api_views.run_change_detection, name="run_detection"),
    path("gemini-analysis/", api_views.gemini_analysis, name="gemini_analysis"),
    path("live-uplink/", api_views.live_uplink_analysis, name="live_uplink_api"),
    path("time-series/", api_views.time_series_analysis, name="time_series_api"),
    path("disaster-analysis/", api_views.disaster_analysis, name="disaster_api"),
    path("live-disasters/", api_views.live_disaster_feed, name="live_disasters"),
    path("historical-disasters/", api_views.get_historical_disasters, name="historical_disasters"),
    path("disaster-day-by-day/", api_views.disaster_day_by_day, name="disaster_day_by_day"),
    path("disaster-day-by-day-stream/", api_views.disaster_day_by_day_stream, name="disaster_day_by_day_stream"),
    path("disaster-day-by-day-ai/", api_views.disaster_day_by_day_ai, name="disaster_day_by_day_ai"),
]
