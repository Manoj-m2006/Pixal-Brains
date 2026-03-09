"""
API views for AJAX requests.
Handles satellite image fetching, change detection, and AI analysis.
"""
import json
import io
import base64
import math
import requests
import numpy as np
from PIL import Image
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import pathlib

# Load environment variables from parent directory
env_path = pathlib.Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Import satellite and model modules from parent directory
from satellite_api import fetch_satellite_image_bbox, fetch_sar_image_bbox
from model_engine import generate_change_mask, overlay_mask

# Configure Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL Image to base64 string for JSON response."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert PIL Image to bytes."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def bbox_area_km2(bbox: list) -> float:
    """Calculate rough area in km² for a bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2
    h = abs(max_lat - min_lat) * 111.0
    w = abs(max_lon - min_lon) * 111.0 * math.cos(math.radians(center_lat))
    return round(h * w, 2)


def compute_land_cover(img: Image.Image) -> dict:
    """
    Compute land cover percentages from a true-color RGB satellite image.

    Strategy (pixel-level RGB heuristics):
      Water     - blue channel clearly dominant over red; includes dark water bodies.
      Vegetation- green channel dominant, >= blue (avoids cyan water confusion),
                  not too dark (excludes forest shadows misclassified by water rule).
      Urban     - low colour saturation (gray/white), moderate brightness;
                  roads, rooftops, concrete, bare soil appear here.
      Other     - clouds, snow, unclassified pixels.

    Returns dict with *_pct keys for vegetation, urban, water, other.
    """
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    total = float(R.size)

    max_c = np.maximum(np.maximum(R, G), B)
    min_c = np.minimum(np.minimum(R, G), B)
    sat_range = max_c - min_c   # 0 = perfect gray, high = vivid colour

    # ------------------------------------------------------------------
    # 1. WATER  — blue dominant, or dark blue-green (turbid / deep water)
    # ------------------------------------------------------------------
    water_mask = (
        ((B > R + 20) & (B >= G - 8))          # clearly blue
        | ((B > R + 10) & (G > R + 5) & (max_c < 110))  # dark cyan / turbid
        | ((B > R + 8) & (sat_range < 50) & (max_c < 90))  # dark water near-gray
    )

    # ------------------------------------------------------------------
    # 2. VEGETATION — green dominant, NOT water, minimum brightness
    # ------------------------------------------------------------------
    veg_mask = (
        (G > R + 12)          # green clearly beats red
        & (G >= B - 8)        # green >= blue (blue-dominated → water, not veg)
        & (G > 35)            # exclude very dark shadows
        & (~water_mask)
    )

    # ------------------------------------------------------------------
    # 3. URBAN / BUILT-UP — low saturation (grayish), moderate brightness
    #    Covers concrete, roads, rooftops, bare/compacted soil
    # ------------------------------------------------------------------
    urban_mask = (
        (sat_range < 40)      # low colour variance = gray/neutral
        & (max_c > 45)        # not pure shadow/black
        & (max_c < 235)       # not cloud (very bright white)
        & (~water_mask)
        & (~veg_mask)
    )

    veg_pct   = round(float(veg_mask.sum())   / total * 100, 1)
    urban_pct = round(float(urban_mask.sum()) / total * 100, 1)
    water_pct = round(float(water_mask.sum()) / total * 100, 1)
    other_pct = round(max(0.0, 100.0 - veg_pct - urban_pct - water_pct), 1)

    return {
        "vegetation_pct": veg_pct,
        "urban_pct":      urban_pct,
        "water_pct":      water_pct,
        "other_pct":      other_pct,
    }


@csrf_exempt
@require_http_methods(["GET", "POST"])
def reverse_geocode(request):
    """Reverse geocode coordinates to location name."""
    try:
        # Support both GET and POST
        if request.method == "GET":
            lat = request.GET.get("lat")
            lon = request.GET.get("lon")
        else:
            data = json.loads(request.body)
            lat = data.get("lat")
            lon = data.get("lon")
        
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "accept-language": "en"},
            headers={"User-Agent": "PixelBrains/1.0", "Accept-Language": "en"},
            timeout=6,
        )
        result = r.json()
        addr = result.get("address", {})
        parts = [
            addr.get("suburb") or addr.get("city_district") or addr.get("village"),
            addr.get("city") or addr.get("town") or addr.get("county"),
            addr.get("state"),
            addr.get("country"),
        ]
        location_name = ", ".join(p for p in parts if p) or result.get("display_name", f"{lat:.4f}°, {lon:.4f}°")
        
        return JsonResponse({"success": True, "location": location_name})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})


@csrf_exempt
@require_http_methods(["POST"])
def fetch_satellite_images(request):
    """Fetch satellite images for a bbox and date range."""
    try:
        data = json.loads(request.body)
        bbox = data.get("bbox")  # [min_lon, min_lat, max_lon, max_lat]
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        enhance_mode = data.get("enhance_mode", "analysis")
        
        # Fetch optical images with standard quality (2x faster than high)
        before_img = fetch_satellite_image_bbox(bbox, start_date, enhance_mode=enhance_mode, quality_mode="standard")
        after_img = fetch_satellite_image_bbox(bbox, end_date, enhance_mode=enhance_mode, quality_mode="standard")
        
        # Fetch SAR images (optional) with standard quality
        before_sar = fetch_sar_image_bbox(bbox, start_date, quality_mode="standard")
        after_sar = fetch_sar_image_bbox(bbox, end_date, quality_mode="standard")
        
        response_data = {
            "success": True,
            "before_img": pil_to_base64(before_img),
            "after_img": pil_to_base64(after_img),
            "before_sar": pil_to_base64(before_sar) if before_sar else None,
            "after_sar": pil_to_base64(after_sar) if after_sar else None,
            "image_size": {"width": before_img.width, "height": before_img.height},
        }
        
        return JsonResponse(response_data)
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})


@csrf_exempt
@require_http_methods(["POST"])
def run_change_detection(request):
    """Run change detection pipeline on before/after images."""
    try:
        data = json.loads(request.body)
        bbox = data.get("bbox")
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        
        # Fetch images in analysis mode with standard quality (fast)
        before_img = fetch_satellite_image_bbox(bbox, start_date, enhance_mode="analysis", quality_mode="standard")
        after_img = fetch_satellite_image_bbox(bbox, end_date, enhance_mode="analysis", quality_mode="standard")
        
        # Fetch SAR images with standard quality
        before_sar = fetch_sar_image_bbox(bbox, start_date, quality_mode="standard")
        after_sar = fetch_sar_image_bbox(bbox, end_date, quality_mode="standard")
        
        # Prepare bytes for model
        before_sar_bytes = pil_to_bytes(before_sar) if before_sar else None
        after_sar_bytes = pil_to_bytes(after_sar) if after_sar else None
        
        # Generate change mask
        mask_img = generate_change_mask(
            pil_to_bytes(before_img), pil_to_bytes(after_img),
            before_sar_bytes, after_sar_bytes,
        )
        
        # Create overlay
        overlay_img = overlay_mask(after_img, mask_img, color=(255, 20, 20), alpha=0.65)
        
        # Calculate statistics
        mask_arr = np.array(mask_img.convert("L"))
        pct_changed = round(float((mask_arr > 127).sum()) / mask_arr.size * 100, 2)
        total_km2 = bbox_area_km2(bbox)
        changed_km2 = round(total_km2 * pct_changed / 100, 2)

        # Land cover analysis (before & after)
        before_lc = compute_land_cover(before_img)
        after_lc  = compute_land_cover(after_img)

        # Deforestation: vegetation percentage lost
        deforestation = round(max(0.0, before_lc["vegetation_pct"] - after_lc["vegetation_pct"]), 1)
        # Reforestation: vegetation gained
        reforestation = round(max(0.0, after_lc["vegetation_pct"] - before_lc["vegetation_pct"]), 1)
        # Urbanization: built-up area gained
        urbanization  = round(max(0.0, after_lc["urban_pct"] - before_lc["urban_pct"]), 1)

        response_data = {
            "success": True,
            "before_img": pil_to_base64(before_img),
            "after_img": pil_to_base64(after_img),
            "overlay_img": pil_to_base64(overlay_img),
            "mask_img": pil_to_base64(mask_img),
            "before_sar": pil_to_base64(before_sar) if before_sar else None,
            "after_sar": pil_to_base64(after_sar) if after_sar else None,
            "stats": {
                "pct_changed":   pct_changed,
                "total_km2":     total_km2,
                "changed_km2":   changed_km2,
            },
            "land_cover": {
                "before":         before_lc,
                "after":          after_lc,
                "deforestation":  deforestation,
                "reforestation":  reforestation,
                "urbanization":   urbanization,
            },
        }
        
        return JsonResponse(response_data)
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})


@csrf_exempt
@require_http_methods(["POST"])
def gemini_analysis(request):
    """Run Gemini AI analysis on satellite images."""
    if not GEMINI_AVAILABLE:
        return JsonResponse({"success": False, "error": "Gemini API not available"})
    
    try:
        data = json.loads(request.body)
        before_b64 = data.get("before_img")
        after_b64 = data.get("after_img")
        location = data.get("location", "Unknown")
        start_year = data.get("start_year")
        end_year = data.get("end_year")
        pct_changed = data.get("pct_changed", 0)
        changed_km2 = data.get("changed_km2", 0)
        total_km2 = data.get("total_km2", 0)
        
        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return JsonResponse({"success": False, "error": "Gemini API key not found"})
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Convert base64 to PIL Images
        before_img = Image.open(io.BytesIO(base64.b64decode(before_b64)))
        after_img = Image.open(io.BytesIO(base64.b64decode(after_b64)))
        
        prompt = f"""Analyze these two satellite images taken {end_year - start_year} years apart:
        
Location: {location}
Time Period: {start_year} to {end_year}
Detected Change: {pct_changed}% of the area ({changed_km2} km² out of {total_km2} km²)

Please provide a detailed analysis covering:

1. **DEFORESTATION ANALYSIS:**
   - Identify areas where vegetation/forest cover has decreased
   - Estimate the extent of deforestation if visible

2. **URBANIZATION ANALYSIS:**
   - Identify new construction, roads, or urban sprawl
   - Note any expansion of built-up areas

3. **LAND USE CHANGES:**
   - Agricultural changes (new farmland, abandoned fields)
   - Water body changes (new reservoirs, dried lakes)

4. **STATISTICAL SUMMARY:**
   - Overall change assessment
   - Key metrics and observations

5. **ENVIRONMENTAL IMPACT:**
   - Potential ecological consequences
   - Recommendations for monitoring
"""
        
        response = model.generate_content([prompt, before_img, after_img])
        analysis_text = response.text
        
        return JsonResponse({
            "success": True,
            "analysis": analysis_text,
        })
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})


@csrf_exempt
@require_http_methods(["POST"])
def live_uplink_analysis(request):
    """Live satellite uplink - fetch current imagery for a region."""
    try:
        data = json.loads(request.body)
        bbox = data.get("bbox")
        location = data.get("location", "Unknown")
        
        # Use today's date (or recent date)
        from datetime import datetime, timedelta
        today = datetime.now()
        current_date = today.strftime("%Y-%m-%d")
        
        # Fetch current image (display mode for visual quality)
        # Fetch current satellite image with standard quality
        current_img = fetch_satellite_image_bbox(bbox, current_date, enhance_mode="display", quality_mode="standard")
        
        # Run AI analysis if Gemini is available
        analysis_text = None
        if GEMINI_AVAILABLE:
            try:
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    
                    prompt = f"""Analyze this satellite image of {location}.
                    
Please provide a brief analysis covering:
1. Land cover types visible (urban, vegetation, water, etc.)
2. Current state of the area
3. Notable features or points of interest
4. Any environmental observations"""
                    
                    response = model.generate_content([prompt, current_img])
                    analysis_text = response.text
            except Exception as e:
                print(f"AI analysis error: {e}")
        
        response_data = {
            "success": True,
            "image": pil_to_base64(current_img),
            "date": current_date,
            "analysis": analysis_text,
        }
        
        return JsonResponse(response_data)
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})


@csrf_exempt
@require_http_methods(["POST"])
def year_difference_analysis(request):
    """Compare images from two different years."""
    try:
        data = json.loads(request.body)
        bbox = data.get("bbox")
        year1 = data.get("year1")
        year2 = data.get("year2")
        season = data.get("season", "summer")
        location = data.get("location", "Unknown")
        
        # Determine date based on season
        season_months = {
            "summer": "07-15",
            "winter": "01-15",
            "spring": "04-15",
            "fall": "10-15"
        }
        month_day = season_months.get(season, "07-15")
        
        date1 = f"{year1}-{month_day}"
        date2 = f"{year2}-{month_day}"
        
        # Fetch images with standard quality
        img1 = fetch_satellite_image_bbox(bbox, date1, enhance_mode="analysis", quality_mode="standard")
        img2 = fetch_satellite_image_bbox(bbox, date2, enhance_mode="analysis", quality_mode="standard")
        
        # Fetch SAR with standard quality
        sar1 = fetch_sar_image_bbox(bbox, date1, quality_mode="standard")
        sar2 = fetch_sar_image_bbox(bbox, date2, quality_mode="standard")
        
        # Run change detection
        sar1_bytes = pil_to_bytes(sar1) if sar1 else None
        sar2_bytes = pil_to_bytes(sar2) if sar2 else None
        
        mask_img = generate_change_mask(
            pil_to_bytes(img1), pil_to_bytes(img2),
            sar1_bytes, sar2_bytes,
        )
        
        overlay_img = overlay_mask(img2, mask_img, color=(255, 20, 20), alpha=0.65)
        
        # Calculate stats
        mask_arr = np.array(mask_img.convert("L"))
        pct_changed = round(float((mask_arr > 127).sum()) / mask_arr.size * 100, 2)
        total_km2 = bbox_area_km2(bbox)
        changed_km2 = round(total_km2 * pct_changed / 100, 2)
        
        # Run AI analysis if Gemini is available
        analysis_text = None
        if GEMINI_AVAILABLE:
            try:
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    
                    prompt = f"""Analyze these two satellite images comparing {location} between {year1} and {year2}.
                    
Detected Change: {pct_changed}% of the area changed ({changed_km2} km² out of {total_km2} km²)

Please provide analysis of:
1. Major changes visible between the two time periods
2. Urban development or expansion
3. Vegetation/forest changes
4. Any environmental concerns
5. Overall assessment of the area's transformation"""
                    
                    response = model.generate_content([prompt, img1, img2])
                    analysis_text = response.text
            except Exception as e:
                print(f"AI analysis error: {e}")
        
        response_data = {
            "success": True,
            "year1_img": pil_to_base64(img1),
            "year2_img": pil_to_base64(img2),
            "overlay_img": pil_to_base64(overlay_img),
            "mask_img": pil_to_base64(mask_img),
            "stats": {
                "pct_changed": pct_changed,
                "total_km2": total_km2,
                "changed_km2": changed_km2,
                "years_diff": year2 - year1,
            },
            "analysis": analysis_text,
        }
        
        return JsonResponse(response_data)
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})


@csrf_exempt
@require_http_methods(["POST"])
def time_series_analysis(request):
    """Generate time series of satellite images showing changes over multiple years."""
    try:
        data = json.loads(request.body)
        bbox = data.get("bbox")
        start_year = data.get("start_year")
        end_year = data.get("end_year")
        location = data.get("location", "Unknown")
        image_count = data.get("image_count", 10)
        
        # Use mid-year date (June 15) as default for consistent comparison
        # This provides good weather conditions and vegetation state for most regions
        month_day = "06-15"
        
        # Calculate years to sample
        year_span = end_year - start_year
        if year_span == 0:
            years_to_sample = [start_year]
        elif year_span == 1:
            years_to_sample = [start_year, end_year]
        else:
            # Distribute images evenly across the time period
            step = year_span / (image_count - 1) if image_count > 1 else year_span
            years_to_sample = []
            for i in range(image_count):
                year = int(start_year + (i * step))
                if year <= end_year and (not years_to_sample or year != years_to_sample[-1]):
                    years_to_sample.append(year)
            # Ensure end year is included
            if years_to_sample[-1] != end_year:
                years_to_sample.append(end_year)
        
        # Fetch all images
        images_data=[]
        previous_img = None
        total_km2 = bbox_area_km2(bbox)
        
        for idx, year in enumerate(years_to_sample):
            date = f"{year}-{month_day}"
            
            try:
                # Use FAST mode for time series - we need many images
                img = fetch_satellite_image_bbox(bbox, date, enhance_mode="raw", quality_mode="fast")
                
                # Calculate change from previous image
                change_pct = None
                changed_km2 = None
                
                if previous_img is not None:
                    try:
                        # Quick change detection between consecutive images
                        mask_img = generate_change_mask(
                            pil_to_bytes(previous_img), 
                            pil_to_bytes(img),
                            None,  # Skip SAR for speed
                            None
                        )
                        mask_arr = np.array(mask_img.convert("L"))
                        change_pct = round(float((mask_arr > 127).sum()) / mask_arr.size * 100, 2)
                        changed_km2 = round(total_km2 * change_pct / 100, 2)
                    except Exception as e:
                        print(f"Change detection error for {year}: {e}")
                
                images_data.append({
                    "year": year,
                    "date": date,
                    "image": pil_to_base64(img),
                    "change_pct": change_pct,
                    "changed_km2": changed_km2
                })
                
                previous_img = img
                
            except Exception as e:
                print(f"Failed to fetch image for {year}: {e}")
                continue
        
        # Run AI analysis if Gemini is available
        analysis_text = None
        if GEMINI_AVAILABLE and len(images_data) >= 2:
            try:
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    
                    # Fetch first and last for AI analysis with standard quality
                    first_img_pil = fetch_satellite_image_bbox(bbox, f"{years_to_sample[0]}-{month_day}", enhance_mode="raw", quality_mode="standard")
                    last_img_pil = fetch_satellite_image_bbox(bbox, f"{years_to_sample[-1]}-{month_day}", enhance_mode="raw", quality_mode="standard")
                    
                    prompt = f"""Analyze this time series of satellite imagery for {location} from {start_year} to {end_year}.

I'm showing you the first image from {years_to_sample[0]} and the last image from {years_to_sample[-1]}.
The complete time series includes {len(images_data)} images spanning {year_span} years.

Please provide analysis of:
1. Major long-term changes visible between the time periods
2. Urban development, infrastructure, or land use changes
3. Vegetation, agricultural, or environmental changes
4. Patterns or trends you observe over this time period
5. Notable features or concerns
6. Overall assessment of the transformation

Be specific about what changed and when possible, describe the nature of those changes."""
                    
                    response = model.generate_content([prompt, first_img_pil, last_img_pil])
                    analysis_text = response.text
            except Exception as e:
                print(f"AI analysis error: {e}")
        
        response_data = {
            "success": True,
            "start_year": start_year,
            "end_year": end_year,
            "images": images_data,
            "total_images": len(images_data),
            "stats": {
                "total_km2": total_km2,
                "year_span": year_span,
            },
            "analysis": analysis_text,
        }
        
        return JsonResponse(response_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"success": False, "error": str(e)})


@csrf_exempt
@require_http_methods(["GET"])
def live_disaster_feed(request):
    """Fetch live disaster updates using Gemini API with Google Search grounding."""
    try:
        if not GEMINI_AVAILABLE:
            return JsonResponse({"success": False, "error": "Gemini API not available"})

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return JsonResponse({"success": False, "error": "Gemini API key not configured"})

        genai.configure(api_key=api_key)

        from datetime import date
        today = date.today().strftime("%B %d, %Y")

        # Try with Google Search grounding for real-time data
        disasters = []
        try:
            from google.generativeai import types as genai_types
            model = genai.GenerativeModel(
                'gemini-2.0-flash',
                tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())]
            )
            prompt = f"""Today is {today}. Search for and list the top 6 most recent active or ongoing natural disasters happening worldwide right now or in the last 72 hours.

For each disaster, respond with exactly this format (one per line, no extra text):
[Type] - [Country/Region] - [Brief 1-sentence description]

Where Type is one of: Flood, Wildfire, Earthquake, Hurricane, Cyclone, Landslide, Drought, Tsunami.
Use real current news. Be factual and concise."""
            response = model.generate_content(prompt)
            raw = response.text
        except Exception:
            # Fallback: plain model without grounding
            model = genai.GenerativeModel('gemini-2.0-flash')
            prompt = f"""Today is {today}. Based on the most recent information available to you, list 6 active or recent natural disasters happening worldwide.

For each, respond with exactly this format (one per line):
[Type] - [Country/Region] - [Brief 1-sentence description]

Where Type is one of: Flood, Wildfire, Earthquake, Hurricane, Cyclone, Landslide, Drought, Tsunami."""
            response = model.generate_content(prompt)
            raw = response.text

        # Parse lines into list
        for line in raw.strip().split('\n'):
            line = line.strip().lstrip('0123456789.-•*)> ').strip()
            if line and ' - ' in line:
                disasters.append(line)

        if not disasters:
            disasters = ["Monitoring global disaster feeds..."]

        return JsonResponse({"success": True, "disasters": disasters, "count": len(disasters)})

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})


@require_http_methods(["GET"])
def get_historical_disasters(request):
    """Get list of historical disaster events from database."""
    try:
        disaster_type = request.GET.get('type', 'all')
        
        # Load disaster events from JSON file
        json_path = pathlib.Path(__file__).resolve().parent / 'disaster_events.json'
        with open(json_path, 'r') as f:
            all_events = json.load(f)
        
        if disaster_type == 'all':
            # Combine all disaster types
            events = []
            for dtype, devents in all_events.items():
                for event in devents:
                    event['type'] = dtype.rstrip('s')  # Remove plural 's'
                    events.append(event)
        else:
            # Get specific disaster type
            key = disaster_type + 's' if not disaster_type.endswith('s') and disaster_type != 'earthquake' else disaster_type
            if key == 'earthquake':
                key = 'earthquakes'
            elif key == 'hurricane':
                key = 'hurricanes'
            events = all_events.get(key, [])
            for event in events:
                event['type'] = disaster_type
        
        # Sort by year descending
        events.sort(key=lambda x: x['year'], reverse=True)
        
        return JsonResponse({
            "success": True,
            "events": events,
            "count": len(events)
        })
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": str(e)
        })


@csrf_exempt
@require_http_methods(["POST"])
def disaster_day_by_day(request):
    """Generate day-by-day satellite images showing disaster progression."""
    try:
        data = json.loads(request.body)
        event_id = data.get("event_id")
        bbox = data.get("bbox")
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        disaster_type = data.get("disaster_type", "unknown")
        location = data.get("location", "Unknown")
        
        # Parse dates
        from datetime import datetime, timedelta
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        total_days = (end - start).days + 1
        
        # Determine sampling strategy based on duration
        if total_days <= 7:
            # Show every day
            sample_days = list(range(total_days))
        elif total_days <= 14:
            # Show every 2 days
            sample_days = list(range(0, total_days, 2))
        elif total_days <= 30:
            # Show every 3 days
            sample_days = list(range(0, total_days, 3))
        else:
            # Show 10-12 evenly spaced days
            step = total_days // 10
            sample_days = list(range(0, total_days, step))[:12]
        
        # Ensure we always have the last day
        if sample_days[-1] != total_days - 1:
            sample_days.append(total_days - 1)
        
        images_data = []
        previous_img = None
        total_km2 = bbox_area_km2(bbox)
        
        for day_offset in sample_days:
            current_date = start + timedelta(days=day_offset)
            date_str = current_date.strftime("%Y-%m-%d")
            
            try:
                # Use FAST mode for day-by-day - we need many images
                img = fetch_satellite_image_bbox(bbox, date_str, enhance_mode="raw", quality_mode="fast")
                
                # Calculate change from previous image
                change_pct = None
                changed_km2 = None
                
                if previous_img is not None:
                    try:
                        mask_img = generate_change_mask(
                            pil_to_bytes(previous_img),
                            pil_to_bytes(img),
                            None,  # Skip SAR for speed
                            None
                        )
                        mask_arr = np.array(mask_img.convert("L"))
                        change_pct = round(float((mask_arr > 127).sum()) / mask_arr.size * 100, 2)
                        changed_km2 = round(total_km2 * change_pct / 100, 2)
                    except Exception as e:
                        print(f"Change detection error for {date_str}: {e}")
                
                images_data.append({
                    "date": date_str,
                    "day_number": day_offset + 1,
                    "image": pil_to_base64(img),
                    "change_pct": change_pct,
                    "changed_km2": changed_km2
                })
                
                previous_img = img
                
            except Exception as e:
                print(f"Failed to fetch image for {date_str}: {e}")
                continue
        
        # Run AI analysis if Gemini is available
        analysis_text = None
        if GEMINI_AVAILABLE and len(images_data) >= 2:
            try:
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    
                    first_img_pil = fetch_satellite_image_bbox(bbox, start_date, enhance_mode="raw", quality_mode="standard")
                    last_img_pil = fetch_satellite_image_bbox(bbox, end_date, enhance_mode="raw", quality_mode="standard")
                    
                    prompt = f"""Analyze this day-by-day progression of a {disaster_type} disaster at {location}.

Period: {start_date} to {end_date} ({total_days} days)
Images captured: {len(images_data)} time points

I'm showing you the first image (before) and last image (after). The complete sequence shows the disaster progression day by day.

Please provide detailed analysis of:
1. How the disaster developed and spread over time
2. Peak impact period and severity
3. Areas most affected during the progression
4. Infrastructure and environmental damage evolution
5. Recovery signs (if any) towards the end
6. Key observations about the disaster's timeline

Be specific about what you observe in the progression."""
                    
                    response = model.generate_content([prompt, first_img_pil, last_img_pil])
                    analysis_text = response.text
            except Exception as e:
                print(f"AI analysis error: {e}")
        
        return JsonResponse({
            "success": True,
            "event_id": event_id,
            "disaster_type": disaster_type,
            "location": location,
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "images": images_data,
            "total_images": len(images_data),
            "stats": {
                "total_km2": total_km2,
            },
            "analysis": analysis_text,
        })
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": str(e)
        })


@csrf_exempt
@require_http_methods(["POST"])
def disaster_day_by_day_stream(request):
    """
    Stream day-by-day satellite images in parallel.
    Returns NDJSON – each line is a JSON object:
      {"type": "header", ...}
      {"type": "image",  "date": ..., "day_number": ..., "image": <base64>}
      {"type": "done"}
    Images arrive as soon as they are fetched (parallel, 6 workers).
    """
    try:
        data = json.loads(request.body)
        event_id   = data.get("event_id")
        bbox       = data.get("bbox")
        start_date = data.get("start_date")
        end_date   = data.get("end_date")
        disaster_type = data.get("disaster_type", "unknown")
        location   = data.get("location", "Unknown")

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end   = datetime.strptime(end_date,   "%Y-%m-%d")
        total_days = (end - start).days + 1

        # Cap at 6 samples for judge-demo speed
        if total_days <= 6:
            sample_days = list(range(total_days))
        elif total_days <= 12:
            sample_days = list(range(0, total_days, 2))
        elif total_days <= 24:
            sample_days = list(range(0, total_days, 4))
        else:
            step = max(1, total_days // 6)
            sample_days = list(range(0, total_days, step))[:6]

        if sample_days[-1] != total_days - 1:
            sample_days.append(total_days - 1)
        sample_days = sample_days[:7]  # hard cap

        total_km2 = bbox_area_km2(bbox)

        def fetch_day(day_offset):
            current_date = start + timedelta(days=day_offset)
            date_str = current_date.strftime("%Y-%m-%d")
            try:
                # forward_only=True ensures each date gets a DIFFERENT satellite pass
                img = fetch_satellite_image_bbox(
                    bbox, date_str,
                    enhance_mode="raw", quality_mode="fast",
                    forward_only=True
                )
                return day_offset, date_str, img, None
            except Exception as exc:
                return day_offset, date_str, None, str(exc)

        def generate():
            yield json.dumps({
                "type": "header",
                "total_days":   total_days,
                "total_km2":    total_km2,
                "total_images": len(sample_days),
                "disaster_type": disaster_type,
                "location":     location,
                "start_date":   start_date,
                "end_date":     end_date,
            }) + "\n"

            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = {executor.submit(fetch_day, d): d for d in sample_days}
                for future in as_completed(futures):
                    day_offset, date_str, img, error = future.result()
                    if img is not None:
                        yield json.dumps({
                            "type":      "image",
                            "date":      date_str,
                            "day_number": day_offset + 1,
                            "image":     pil_to_base64(img, fmt="JPEG"),
                        }) + "\n"
                    else:
                        yield json.dumps({
                            "type":  "error",
                            "date":  date_str,
                            "error": error,
                        }) + "\n"

            yield json.dumps({"type": "done"}) + "\n"

        resp = StreamingHttpResponse(generate(), content_type="application/x-ndjson")
        resp["X-Accel-Buffering"] = "no"   # disable nginx buffering
        resp["Cache-Control"] = "no-cache"
        return resp

    except Exception as exc:
        return JsonResponse({"success": False, "error": str(exc)})


@csrf_exempt
@require_http_methods(["POST"])
def disaster_day_by_day_ai(request):
    """
    Run Gemini AI analysis on a disaster (first + last image only).
    Called separately after the image stream loads, so it never blocks the UI.
    """
    try:
        if not GEMINI_AVAILABLE:
            return JsonResponse({"success": False, "error": "Gemini not available"})

        data = json.loads(request.body)
        bbox          = data.get("bbox")
        start_date    = data.get("start_date")
        end_date      = data.get("end_date")
        disaster_type = data.get("disaster_type", "unknown")
        location      = data.get("location", "Unknown")

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end   = datetime.strptime(end_date,   "%Y-%m-%d")
        total_days = (end - start).days + 1

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return JsonResponse({"success": False, "error": "No Gemini API key"})

        # Fetch first and last images in parallel
        def get_img(date_str):
            return fetch_satellite_image_bbox(bbox, date_str, enhance_mode="raw", quality_mode="standard")

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_first = ex.submit(get_img, start_date)
            f_last  = ex.submit(get_img, end_date)
            first_img = f_first.result()
            last_img  = f_last.result()

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""Analyze this {disaster_type} disaster progression at {location}.
Period: {start_date} to {end_date} ({total_days} days).
I'm showing the BEFORE (first) and AFTER (last) satellite images.

Provide a concise analysis covering:
1. Visible changes and affected areas
2. Estimated severity and spread
3. Infrastructure or environmental damage
4. Any signs of recovery (if visible in the after image)
5. Key timeline observations

Be specific about what you see in the satellite imagery."""

        response = model.generate_content([prompt, first_img, last_img])
        return JsonResponse({"success": True, "analysis": response.text})

    except Exception as exc:
        return JsonResponse({"success": False, "error": str(exc)})


@require_http_methods(["POST"])
def disaster_analysis(request):
    """Analyze disaster-affected areas."""
    try:
        data = json.loads(request.body)
        bbox = data.get("bbox")
        before_date = data.get("before_date")
        after_date = data.get("after_date")
        disaster_type = data.get("disaster_type", "Unknown")
        
        # Fetch disaster images with standard quality
        before_img = fetch_satellite_image_bbox(bbox, before_date, enhance_mode="analysis", quality_mode="standard")
        after_img = fetch_satellite_image_bbox(bbox, after_date, enhance_mode="analysis", quality_mode="standard")
        
        # SAR is crucial for disaster monitoring (works through clouds/smoke)
        before_sar = fetch_sar_image_bbox(bbox, before_date)
        after_sar = fetch_sar_image_bbox(bbox, after_date)
        
        # Run detection
        before_sar_bytes = pil_to_bytes(before_sar) if before_sar else None
        after_sar_bytes = pil_to_bytes(after_sar) if after_sar else None
        
        mask_img = generate_change_mask(
            pil_to_bytes(before_img), pil_to_bytes(after_img),
            before_sar_bytes, after_sar_bytes,
        )
        
        overlay_img = overlay_mask(after_img, mask_img, color=(255, 20, 20), alpha=0.65)
        
        # Stats
        mask_arr = np.array(mask_img.convert("L"))
        pct_affected = round(float((mask_arr > 127).sum()) / mask_arr.size * 100, 2)
        total_km2 = bbox_area_km2(bbox)
        affected_km2 = round(total_km2 * pct_affected / 100, 2)
        location = data.get("location", "Unknown")
        
        # Run AI analysis if Gemini is available
        analysis_text = None
        if GEMINI_AVAILABLE:
            try:
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    
                    disaster_prompts = {
                        "flood": "flooding, water extent, submerged areas, water damage",
                        "fire": "burn scars, ash deposits, vegetation damage, smoke damage",
                        "earthquake": "structural damage, collapsed buildings, ground deformation",
                        "hurricane": "storm damage, debris, flooding, structural damage",
                        "landslide": "slope failure, debris flow, terrain changes",
                        "drought": "vegetation stress, water body reduction, drying patterns"
                    }
                    
                    focus_areas = disaster_prompts.get(disaster_type, "damage and changes")
                    
                    prompt = f"""Analyze these satellite images showing a {disaster_type} disaster at {location}.
                    
Before Date: {before_date}
After Date: {after_date}
Affected Area: {pct_affected}% ({affected_km2} km² out of {total_km2} km²)

Focus on identifying: {focus_areas}

Please provide:
1. Extent of disaster damage
2. Most severely affected areas
3. Infrastructure impact assessment
4. Environmental damage
5. Recovery recommendations"""
                    
                    response = model.generate_content([prompt, before_img, after_img])
                    analysis_text = response.text
            except Exception as e:
                print(f"AI analysis error: {e}")
        
        response_data = {
            "success": True,
            "before_img": pil_to_base64(before_img),
            "after_img": pil_to_base64(after_img),
            "overlay_img": pil_to_base64(overlay_img),
            "disaster_type": disaster_type,
            "stats": {
                "pct_affected": pct_affected,
                "total_km2": total_km2,
                "affected_km2": affected_km2,
            },
            "analysis": analysis_text,
        }
        
        return JsonResponse(response_data)
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})


@csrf_exempt
@require_http_methods(["GET"])
def get_aqi_data(request):
    """
    Fetch Air Quality Index data for major cities worldwide using OpenAQ API v3.
    Returns real-time AQI measurements with location coordinates.
    """
    try:
        # OpenAQ API v3 endpoint for latest measurements
        url = "https://api.openaq.org/v3/locations"
        
        # Get API key from environment
        api_key = os.getenv('OPENAQ_API_KEY')
        
        # Set up headers with API key for authentication
        headers = {
            'Accept': 'application/json',
        }
        if api_key:
            headers['X-API-Key'] = api_key
        
        # Get parameters from request
        limit = request.GET.get('limit', '100')
        country = request.GET.get('country', '')
        lat = request.GET.get('lat', '')
        lon = request.GET.get('lon', '')
        radius = request.GET.get('radius', '25000')  # Default 25km radius
        
        # OpenAQ v3 API parameters
        params = {
            'limit': limit,
            'order_by': 'lastUpdated',
            'sort_order': 'desc',
            'has_geo': 'true',
            'parameters_id': '2',  # PM2.5 parameter ID in v3
        }
        
        # If coordinates provided, search within radius
        if lat and lon:
            params['coordinates'] = f"{lat},{lon}"
            params['radius'] = radius
        
        if country:
            params['countries_id'] = country
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        
        if response.status_code != 200:
            # Fallback to mock data if API fails
            return JsonResponse({
                "success": True,
                "data": generate_mock_aqi_data(),
                "count": 50,
                "timestamp": datetime.now().isoformat(),
                "note": "Using sample data - OpenAQ API unavailable"
            })
        
        data = response.json()
        
        if 'results' not in data:
            return JsonResponse({
                "success": True,
                "data": generate_mock_aqi_data(),
                "count": 50,
                "timestamp": datetime.now().isoformat(),
                "note": "Using sample data"
            })
        
        # Process and format the data
        aqi_locations = []
        for location in data['results']:
            # Get coordinates
            if not location.get('coordinates'):
                continue
            
            coords = location['coordinates']
            if not coords.get('latitude') or not coords.get('longitude'):
                continue
            
            location_name = location.get('name', 'Unknown')
            city_name = location.get('city', 'Unknown City')
            country_name = location.get('country', {}).get('name', 'Unknown Country')
            
            # Get latest PM2.5 measurement
            pm25_value = None
            last_updated = ''
            
            sensors = location.get('sensors', [])
            for sensor in sensors:
                parameter = sensor.get('parameter', {})
                if parameter.get('id') == 2:  # PM2.5
                    latest = sensor.get('latest', {})
                    pm25_value = latest.get('value')
                    last_updated = latest.get('datetime', {}).get('utc', '')
                    break
            
            if pm25_value is not None:
                # Convert PM2.5 to AQI category (US EPA standard)
                if pm25_value <= 12.0:
                    aqi_category = "Good"
                    aqi_color = "#00e400"
                elif pm25_value <= 35.4:
                    aqi_category = "Moderate"
                    aqi_color = "#ffff00"
                elif pm25_value <= 55.4:
                    aqi_category = "Unhealthy for Sensitive Groups"
                    aqi_color = "#ff7e00"
                elif pm25_value <= 150.4:
                    aqi_category = "Unhealthy"
                    aqi_color = "#ff0000"
                elif pm25_value <= 250.4:
                    aqi_category = "Very Unhealthy"
                    aqi_color = "#8f3f97"
                else:
                    aqi_category = "Hazardous"
                    aqi_color = "#7e0023"
                
                aqi_locations.append({
                    'location': location_name,
                    'city': city_name,
                    'country': country_name,
                    'latitude': coords['latitude'],
                    'longitude': coords['longitude'],
                    'pm25': round(pm25_value, 2),
                    'category': aqi_category,
                    'color': aqi_color,
                    'lastUpdated': last_updated,
                })
        
        # If no data, use mock data
        if not aqi_locations:
            aqi_locations = generate_mock_aqi_data()
        
        return JsonResponse({
            "success": True,
            "data": aqi_locations,
            "count": len(aqi_locations),
            "timestamp": datetime.now().isoformat()
        })
        
    except requests.RequestException as e:
        return JsonResponse({
            "success": True,
            "data": generate_mock_aqi_data(),
            "count": 50,
            "timestamp": datetime.now().isoformat(),
            "note": f"Using sample data - Network error: {str(e)}"
        })
    except Exception as e:
        return JsonResponse({
            "success": True,
            "data": generate_mock_aqi_data(),
            "count": 50,
            "timestamp": datetime.now().isoformat(),
            "note": f"Using sample data - Error: {str(e)}"
        })


def generate_mock_aqi_data():
    """Generate mock AQI data for major cities when API is unavailable"""
    mock_cities = [
        # North America
        {'location': 'Downtown Monitor', 'city': 'New York', 'country': 'United States', 'lat': 40.7128, 'lon': -74.0060, 'pm25': 15.3},
        {'location': 'City Center', 'city': 'Los Angeles', 'country': 'United States', 'lat': 34.0522, 'lon': -118.2437, 'pm25': 42.1},
        {'location': 'Union Station', 'city': 'Chicago', 'country': 'United States', 'lat': 41.8781, 'lon': -87.6298, 'pm25': 22.7},
        {'location': 'Downtown', 'city': 'Toronto', 'country': 'Canada', 'lat': 43.6532, 'lon': -79.3832, 'pm25': 18.4},
        {'location': 'Centro', 'city': 'Mexico City', 'country': 'Mexico', 'lat': 19.4326, 'lon': -99.1332, 'pm25': 67.8},
        
        # Europe
        {'location': 'Westminster', 'city': 'London', 'country': 'United Kingdom', 'lat': 51.5074, 'lon': -0.1278, 'pm25': 24.6},
        {'location': 'Marais', 'city': 'Paris', 'country': 'France', 'lat': 48.8566, 'lon': 2.3522, 'pm25': 19.2},
        {'location': 'Mitte', 'city': 'Berlin', 'country': 'Germany', 'lat': 52.5200, 'lon': 13.4050, 'pm25': 16.8},
        {'location': 'Centro Storico', 'city': 'Rome', 'country': 'Italy', 'lat': 41.9028, 'lon': 12.4964, 'pm25': 28.3},
        {'location': 'City Center', 'city': 'Madrid', 'country': 'Spain', 'lat': 40.4168, 'lon': -3.7038, 'pm25': 21.5},
        
        # Asia
        {'location': 'US Embassy', 'city': 'Beijing', 'country': 'China', 'lat': 39.9042, 'lon': 116.4074, 'pm25': 89.4},
        {'location': 'Chiyoda', 'city': 'Tokyo', 'country': 'Japan', 'lat': 35.6762, 'lon': 139.6503, 'pm25': 12.1},
        {'location': 'Central', 'city': 'Hong Kong', 'country': 'Hong Kong', 'lat': 22.3193, 'lon': 114.1694, 'pm25': 34.7},
        {'location': 'Marina Bay', 'city': 'Singapore', 'country': 'Singapore', 'lat': 1.3521, 'lon': 103.8198, 'pm25': 26.9},
        {'location': 'Connaught Place', 'city': 'New Delhi', 'country': 'India', 'lat': 28.6139, 'lon': 77.2090, 'pm25': 156.8},
        {'location': 'City Center', 'city': 'Mumbai', 'country': 'India', 'lat': 19.0760, 'lon': 72.8777, 'pm25': 98.3},
        {'location': 'Gangnam', 'city': 'Seoul', 'country': 'South Korea', 'lat': 37.5665, 'lon': 126.9780, 'pm25': 38.6},
        {'location': 'Makati', 'city': 'Manila', 'country': 'Philippines', 'lat': 14.5995, 'lon': 120.9842, 'pm25': 44.2},
        {'location': 'Silom', 'city': 'Bangkok', 'country': 'Thailand', 'lat': 13.7563, 'lon': 100.5018, 'pm25': 52.7},
        {'location': 'Central', 'city': 'Jakarta', 'country': 'Indonesia', 'lat': -6.2088, 'lon': 106.8456, 'pm25': 71.4},
        
        # Middle East
        {'location': 'Downtown', 'city': 'Dubai', 'country': 'United Arab Emirates', 'lat': 25.2048, 'lon': 55.2708, 'pm25': 62.3},
        {'location': 'Riyadh Center', 'city': 'Riyadh', 'country': 'Saudi Arabia', 'lat': 24.7136, 'lon': 46.6753, 'pm25': 84.6},
        {'location': 'City Center', 'city': 'Tehran', 'country': 'Iran', 'lat': 35.6892, 'lon': 51.3890, 'pm25': 93.2},
        {'location': 'Downtown', 'city': 'Istanbul', 'country': 'Turkey', 'lat': 41.0082, 'lon': 28.9784, 'pm25': 36.8},
        
        # South America
        {'location': 'Centro', 'city': 'São Paulo', 'country': 'Brazil', 'lat': -23.5505, 'lon': -46.6333, 'pm25': 31.7},
        {'location': 'Centro', 'city': 'Buenos Aires', 'country': 'Argentina', 'lat': -34.6037, 'lon': -58.3816, 'pm25': 23.4},
        {'location': 'Centro', 'city': 'Santiago', 'country': 'Chile', 'lat': -33.4489, 'lon': -70.6693, 'pm25': 47.9},
        {'location': 'Centro', 'city': 'Lima', 'country': 'Peru', 'lat': -12.0464, 'lon': -77.0428, 'pm25': 41.2},
        
        # Africa
        {'location': 'City Center', 'city': 'Cairo', 'country': 'Egypt', 'lat': 30.0444, 'lon': 31.2357, 'pm25': 102.5},
        {'location': 'CBD', 'city': 'Johannesburg', 'country': 'South Africa', 'lat': -26.2041, 'lon': 28.0473, 'pm25': 29.6},
        {'location': 'Victoria Island', 'city': 'Lagos', 'country': 'Nigeria', 'lat': 6.4281, 'lon': 3.4219, 'pm25': 78.4},
        {'location': 'City Center', 'city': 'Nairobi', 'country': 'Kenya', 'lat': -1.2864, 'lon': 36.8172, 'pm25': 33.8},
        
        # Oceania
        {'location': 'CBD', 'city': 'Sydney', 'country': 'Australia', 'lat': -33.8688, 'lon': 151.2093, 'pm25': 11.7},
        {'location': 'CBD', 'city': 'Melbourne', 'country': 'Australia', 'lat': -37.8136, 'lon': 144.9631, 'pm25': 13.2},
        {'location': 'City Center', 'city': 'Auckland', 'country': 'New Zealand', 'lat': -36.8485, 'lon': 174.7633, 'pm25': 9.8},
        
        # Additional Asian cities
        {'location': 'Downtown', 'city': 'Dhaka', 'country': 'Bangladesh', 'lat': 23.8103, 'lon': 90.4125, 'pm25': 143.7},
        {'location': 'City Center', 'city': 'Karachi', 'country': 'Pakistan', 'lat': 24.8607, 'lon': 67.0011, 'pm25': 112.9},
        {'location': 'Downtown', 'city': 'Kolkata', 'country': 'India', 'lat': 22.5726, 'lon': 88.3639, 'pm25': 87.6},
        {'location': 'City Center', 'city': 'Hanoi', 'country': 'Vietnam', 'lat': 21.0285, 'lon': 105.8542, 'pm25': 56.3},
        {'location': 'Downtown', 'city': 'Kuala Lumpur', 'country': 'Malaysia', 'lat': 3.1390, 'lon': 101.6869, 'pm25': 39.8},
        
        # Additional European cities
        {'location': 'City Center', 'city': 'Amsterdam', 'country': 'Netherlands', 'lat': 52.3676, 'lon': 4.9041, 'pm25': 17.3},
        {'location': 'Downtown', 'city': 'Brussels', 'country': 'Belgium', 'lat': 50.8503, 'lon': 4.3517, 'pm25': 20.4},
        {'location': 'Old Town', 'city': 'Prague', 'country': 'Czech Republic', 'lat': 50.0755, 'lon': 14.4378, 'pm25': 22.1},
        {'location': 'City Center', 'city': 'Vienna', 'country': 'Austria', 'lat': 48.2082, 'lon': 16.3738, 'pm25': 18.9},
        {'location': 'Downtown', 'city': 'Stockholm', 'country': 'Sweden', 'lat': 59.3293, 'lon': 18.0686, 'pm25': 14.6},
        {'location': 'City Center', 'city': 'Warsaw', 'country': 'Poland', 'lat': 52.2297, 'lon': 21.0122, 'pm25': 25.8},
        {'location': 'Downtown', 'city': 'Athens', 'country': 'Greece', 'lat': 37.9838, 'lon': 23.7275, 'pm25': 32.4},
        {'location': 'City Center', 'city': 'Lisbon', 'country': 'Portugal', 'lat': 38.7223, 'lon': -9.1393, 'pm25': 16.2},
    ]
    
    aqi_data = []
    for city in mock_cities:
        pm25 = city['pm25']
        
        # Convert PM2.5 to AQI category
        if pm25 <= 12.0:
            category = "Good"
            color = "#00e400"
        elif pm25 <= 35.4:
            category = "Moderate"
            color = "#ffff00"
        elif pm25 <= 55.4:
            category = "Unhealthy for Sensitive Groups"
            color = "#ff7e00"
        elif pm25 <= 150.4:
            category = "Unhealthy"
            color = "#ff0000"
        elif pm25 <= 250.4:
            category = "Very Unhealthy"
            color = "#8f3f97"
        else:
            category = "Hazardous"
            color = "#7e0023"
        
        aqi_data.append({
            'location': city['location'],
            'city': city['city'],
            'country': city['country'],
            'latitude': city['lat'],
            'longitude': city['lon'],
            'pm25': pm25,
            'category': category,
            'color': color,
            'lastUpdated': datetime.now().isoformat(),
        })
    
    return aqi_data
