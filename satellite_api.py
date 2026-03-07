"""
Copernicus Data Space Ecosystem - Sentinel Hub Satellite API
Fetches Sentinel-2 (optical) and Sentinel-1 (SAR) imagery using the
Sentinel Hub Process API via OAuth2 credentials stored in .env.

Credentials required in .env:
  COPERNICUS_CLIENT_ID=sh-xxxxxxxx-...
  COPERNICUS_CLIENT_SECRET=...
"""

import os
import io
import math
import hashlib
import datetime
import time as _time
import requests
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from io import BytesIO
from dotenv import load_dotenv

# -- Load env -----------------------------------------------------------------
load_dotenv()

# -- Constants ----------------------------------------------------------------
_TOKEN_URL   = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# -- Cache --------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_key(bbox_coords: list, date_string: str, img_type: str) -> str:
    rounded_bbox = [round(x, 3) for x in bbox_coords]
    key_str = f"{rounded_bbox}_{date_string}_{img_type}"
    return hashlib.md5(key_str.encode()).hexdigest()

def _get_cached_image(cache_key: str):
    import time
    for ext in ["webp", "jpg", "png"]:
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.{ext}")
        if os.path.exists(cache_path):
            age_days = (time.time() - os.path.getmtime(cache_path)) / 86400
            if age_days < 90:
                print(f"   Cache hit ({ext}, {age_days:.1f}d old)")
                return Image.open(cache_path).copy()
            else:
                os.remove(cache_path)
    return None

def _save_to_cache(cache_key: str, img: Image.Image):
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.webp")
    img.convert("RGB").save(cache_path, "WEBP", quality=85)

# -- OAuth2 token (in-process cache) -----------------------------------------
_token_cache = {"token": None, "expires_at": 0.0}

def _get_access_token() -> str:
    import time
    if time.time() < _token_cache["expires_at"] - 30 and _token_cache["token"]:
        return _token_cache["token"]
    client_id     = os.getenv("COPERNICUS_CLIENT_ID", "")
    client_secret = os.getenv("COPERNICUS_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise RuntimeError("COPERNICUS_CLIENT_ID / COPERNICUS_CLIENT_SECRET not set in .env")
    resp = requests.post(
        _TOKEN_URL,
        data={"grant_type": "client_credentials",
              "client_id": client_id,
              "client_secret": client_secret},
        timeout=20,
    )
    resp.raise_for_status()
    js = resp.json()
    import time as t2
    _token_cache["token"]      = js["access_token"]
    _token_cache["expires_at"] = t2.time() + int(js.get("expires_in", 3600))
    print("   Copernicus token obtained")
    return _token_cache["token"]

# -- Adaptive size ------------------------------------------------------------
def _adaptive_size(bbox_coords: list, mode: str = "standard") -> tuple:
    minx, miny, maxx, maxy = bbox_coords
    w_deg = maxx - minx
    h_deg = maxy - miny
    ar = w_deg / h_deg if h_deg > 0 else 1.0
    base = {"fast": 512, "high": 1024}.get(mode, 768)
    if ar > 1:
        width, height = base, max(256, int(base / ar))
    else:
        width, height = max(256, int(base * ar)), base
    return min(2500, width), min(2500, height)

# -- Image enhancement --------------------------------------------------------
def _enhance_image_quality(img: Image.Image, scale_factor: float = 1.0, mode: str = "display") -> Image.Image:
    arr = np.array(img.convert("RGB"))
    if mode in ("display", "analysis"):
        arr = cv2.bilateralFilter(arr, 5, 50, 50)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        arr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    if scale_factor > 1.0:
        new_w = int(arr.shape[1] * scale_factor)
        new_h = int(arr.shape[0] * scale_factor)
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(arr)

# -- Evalscripts --------------------------------------------------------------
_S2_EVALSCRIPT = """
//VERSION=3
function setup() {
  return { input: ["B04","B03","B02","dataMask"],
           output: { bands: 4, sampleType: "UINT8" } };
}
function evaluatePixel(s) {
  const gain = 3.5;
  return [ Math.min(255, s.B04 * gain * 255),
           Math.min(255, s.B03 * gain * 255),
           Math.min(255, s.B02 * gain * 255),
           s.dataMask ? 255 : 0 ];
}
"""

_SAR_EVALSCRIPT = """
//VERSION=3
function setup() {
  return { input: ["VV","VH","dataMask"],
           output: { bands: 4, sampleType: "UINT8" } };
}
function evaluatePixel(s) {
  const vv = Math.sqrt(Math.max(0, s.VV));
  const vh = Math.sqrt(Math.max(0, s.VH));
  const ratio = Math.min(1, vv / (vh + 1e-6));
  return [ Math.min(255, vv * 800),
           Math.min(255, vh * 800),
           Math.min(255, ratio * 200),
           s.dataMask ? 255 : 0 ];
}
"""

# -- Core fetch helpers -------------------------------------------------------
def _sentinel2_process(bbox_coords, date_string, width, height, window_days=60, forward_only=False):
    target    = datetime.datetime.strptime(date_string, "%Y-%m-%d")
    if forward_only:
        # Look FORWARD from this date so consecutive dates get different acquisitions
        time_from = target.strftime("%Y-%m-%dT00:00:00Z")
        time_to   = (target + datetime.timedelta(days=window_days)).strftime("%Y-%m-%dT23:59:59Z")
    else:
        time_from = (target - datetime.timedelta(days=window_days)).strftime("%Y-%m-%dT00:00:00Z")
        time_to   = (target + datetime.timedelta(days=window_days)).strftime("%Y-%m-%dT23:59:59Z")
    minx, miny, maxx, maxy = bbox_coords
    payload = {
        "input": {
            "bounds": {
                "bbox": [minx, miny, maxx, maxy],
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {"from": time_from, "to": time_to},
                    "maxCloudCoverage": 80,
                    "mosaickingOrder": "leastCC" if not forward_only else "mostRecent",
                },
            }],
        },
        "output": {
            "width": width, "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}],
        },
        "evalscript": _S2_EVALSCRIPT,
    }
    token   = _get_access_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp    = requests.post(_PROCESS_URL, json=payload, headers=headers, timeout=120)
    if resp.status_code == 204:
        raise ValueError("No Sentinel-2 imagery for this region/period (HTTP 204)")
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    if np.array(img).mean() < 3:
        raise ValueError("Returned image is all-black (no data)")
    return img


def _sentinel1_process(bbox_coords, date_string, width, height, window_days=30):
    target    = datetime.datetime.strptime(date_string, "%Y-%m-%d")
    time_from = (target - datetime.timedelta(days=window_days)).strftime("%Y-%m-%dT00:00:00Z")
    time_to   = (target + datetime.timedelta(days=window_days)).strftime("%Y-%m-%dT23:59:59Z")
    minx, miny, maxx, maxy = bbox_coords
    payload = {
        "input": {
            "bounds": {
                "bbox": [minx, miny, maxx, maxy],
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "timeRange": {"from": time_from, "to": time_to},
                    "acquisitionMode": "IW",
                    "polarization": "DV",
                    "mosaickingOrder": "mostRecent",
                },
                "processing": {"orthorectify": True, "backCoeff": "SIGMA0_ELLIPSOID"},
            }],
        },
        "output": {
            "width": width, "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}],
        },
        "evalscript": _SAR_EVALSCRIPT,
    }
    token   = _get_access_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp    = requests.post(_PROCESS_URL, json=payload, headers=headers, timeout=120)
    if resp.status_code == 204:
        return None
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    if np.array(img).mean() < 3:
        return None
    return img

# -- Public API ---------------------------------------------------------------
def fetch_satellite_image_bbox(
    bbox_coords: list,
    date_string: str,
    enhance_mode: str = "display",
    quality_mode: str = "standard",
    forward_only: bool = False,
) -> Image.Image:
    """
    Fetch Sentinel-2 L2A optical satellite imagery for a bounding box.

    Args:
        bbox_coords:  [minx, miny, maxx, maxy] WGS-84
        date_string:  "YYYY-MM-DD"
        enhance_mode: "raw" | "display" | "analysis"
        quality_mode: "fast" | "standard" | "high"

    Returns:
        PIL Image (RGB)
    """
    print(f"\n[S2] Copernicus Sentinel-2 fetch")
    print(f"   BBox: {bbox_coords}  Date: {date_string}  mode: {enhance_mode}/{quality_mode}")

    ck = _cache_key(bbox_coords, date_string, f"s2_{enhance_mode}_{quality_mode}{'_fwd' if forward_only else ''}")
    cached = _get_cached_image(ck)
    if cached:
        return cached

    width, height = _adaptive_size(bbox_coords, quality_mode)
    window        = 20 if forward_only else (45 if quality_mode == "fast" else 90)

    img = _sentinel2_process(bbox_coords, date_string, width, height, window_days=window, forward_only=forward_only)
    print(f"   OK Sentinel-2 {img.size}")

    if enhance_mode != "raw":
        img = _enhance_image_quality(img, scale_factor=1.0, mode=enhance_mode)

    _save_to_cache(ck, img)
    return img


def fetch_sar_image_bbox(
    bbox_coords: list,
    date_string: str,
    quality_mode: str = "standard",
):
    """
    Fetch Sentinel-1 SAR imagery.  Returns None if unavailable.
    """
    print(f"\n[SAR] Copernicus Sentinel-1 fetch")
    print(f"   BBox: {bbox_coords}  Date: {date_string}")

    ck = _cache_key(bbox_coords, date_string, f"sar_{quality_mode}")
    cached = _get_cached_image(ck)
    if cached:
        return cached

    width, height = _adaptive_size(bbox_coords, quality_mode)
    try:
        img = _sentinel1_process(bbox_coords, date_string, width, height)
        if img is None:
            print("   No SAR data available")
            return None
        print(f"   OK SAR {img.size}")
        _save_to_cache(ck, img)
        return img
    except Exception as exc:
        print(f"   SAR fetch failed: {exc}")
        return None


def fetch_satellite_image(lat: float, lon: float, date_string: str, zoom: float = 0.15) -> Image.Image:
    """Convenience: fetch image for a point location."""
    bbox = [lon - zoom/2, lat - zoom/2, lon + zoom/2, lat + zoom/2]
    return fetch_satellite_image_bbox(bbox, date_string, "display", "standard")


# -- Startup check ------------------------------------------------------------
print("=" * 70)
print("COPERNICUS DATA SPACE ECOSYSTEM - SENTINEL HUB SATELLITE API")

_cid = os.getenv("COPERNICUS_CLIENT_ID", "")
_cs  = os.getenv("COPERNICUS_CLIENT_SECRET", "")

if _cid and _cs:
    try:
        _get_access_token()
        print(f"   OK  Authenticated  client={_cid[:24]}...")
        print("   Sentinel-2 optical + Sentinel-1 SAR ready")
    except Exception as _e:
        print(f"   WARNING  Token fetch failed: {_e}")
        print("   Check COPERNICUS_CLIENT_ID / COPERNICUS_CLIENT_SECRET in .env")
else:
    print("   WARNING  No Copernicus credentials found in .env")
    print("   Set COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET")

print("=" * 70)


if __name__ == "__main__":
    print("\nTEST: Sentinel-2 optical")
    try:
        img = fetch_satellite_image_bbox([77.1, 28.5, 77.3, 28.7], "2024-06-01", "display", "fast")
        img.save("test_s2_copernicus.png")
        print(f"Saved test_s2_copernicus.png {img.size}")
    except Exception as ex:
        print(f"FAIL {ex}")
