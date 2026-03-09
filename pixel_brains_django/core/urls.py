"""
Core app URL configuration for page views.
"""
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("change-detection/", views.change_detection, name="change_detection"),
    path("analysis/", views.analysis_results, name="analysis_results"),
    path("time-series/", views.time_series_analysis, name="time_series"),
    path("disasters/", views.disasters, name="disasters"),
    path("air-quality/", views.air_quality, name="air_quality"),
]
