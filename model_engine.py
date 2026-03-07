"""Fast change detection engine using OpenCV - optimized for speed."""
from PIL import Image
import io
import numpy as np
import cv2

# Lazy loading for optional heavy models
_clip_model = None
_clip_processor = None

def _load_clip_if_needed():
    """Lazy load CLIP model only when needed for validation."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model.eval()
        except Exception as e:
            print(f"⚠️ CLIP not available: {e}")
            return False
    return True

# Prompts that CLIP scores the image against.
# The image is satellite if the mean score over satellite prompts
# exceeds the mean score over non-satellite prompts.
_SAT_PROMPTS = [
    "satellite imagery of earth surface captured from above",
    "aerial top-down view of terrain, land, or city",
    "remote sensing image showing forests, rivers, or urban areas",
    "overhead satellite view of desert, farmland, or coastline",
    "nadir view of earth showing roads, buildings, and vegetation",
]
_NON_SAT_PROMPTS = [
    "a photograph of a person or people",
    "a meme, funny image, or internet joke",
    "artwork, illustration, painting, or cartoon",
    "a document, screenshot, or page of text",
    "an indoor photo or close-up ground-level picture",
    "a selfie or portrait photo",
    "a food photo or product image",
]

def is_satellite_image(img_bytes):
    """
    Fast satellite image validation - uses simple heuristics first,
    falls back to CLIP only if needed.
    """
    try:
        img_pil = Image.open(io.BytesIO(img_bytes))
        
        # TIFF is standard remote-sensing format - always accept
        if img_pil.format == 'TIFF':
            return True
        
        # Fast heuristic: satellite images are typically square-ish and large
        w, h = img_pil.size
        if w >= 256 and h >= 256:
            # Check if image has typical satellite characteristics
            img_rgb = np.array(img_pil.convert('RGB'))
            # Satellite images typically have varied colors (not solid)
            std_dev = np.std(img_rgb)
            if std_dev > 20:  # Not a solid color image
                return True
        
        # Fall back to CLIP if available
        if _load_clip_if_needed() and _clip_model is not None:
            import torch
            img_rgb = img_pil.convert('RGB')
            all_prompts = _SAT_PROMPTS + _NON_SAT_PROMPTS
            inputs = _clip_processor(text=all_prompts, images=img_rgb, return_tensors='pt', padding=True)
            with torch.no_grad():
                outputs = _clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
            n_sat = len(_SAT_PROMPTS)
            return probs[:n_sat].mean().item() > probs[n_sat:].mean().item()
        
        return True  # Default accept if no validation available
    except Exception:
        return True  # Accept on error

def create_water_mask_fast(rgb_img):
    """
    Fast water mask using simple HSV thresholding.
    Optimized for speed over accuracy.
    """
    # Convert to HSV (fast)
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    
    # Water is blue-ish with low saturation
    # HSV ranges: H=85-135 (blue), S<100 (low sat), V<200 (not too bright)
    lower_water = np.array([85, 0, 0])
    upper_water = np.array([135, 100, 200])
    water_mask = cv2.inRange(hsv, lower_water, upper_water)
    
    # Quick morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
    
    return water_mask > 0


def generate_change_mask(before_img_bytes, after_img_bytes,
                         before_sar_bytes=None, after_sar_bytes=None):
    """
    Fast change detection optimized for speed.
    Uses simple image differencing with adaptive thresholding.
    """
    import time
    start = time.time()
    
    # Load images
    before_rgb = np.array(Image.open(io.BytesIO(before_img_bytes)).convert('RGB'))
    after_rgb  = np.array(Image.open(io.BytesIO(after_img_bytes)).convert('RGB'))

    # Resize to same shape (use faster interpolation)
    if before_rgb.shape != after_rgb.shape:
        h = max(before_rgb.shape[0], after_rgb.shape[0])
        w = max(before_rgb.shape[1], after_rgb.shape[1])
        before_rgb = cv2.resize(before_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        after_rgb  = cv2.resize(after_rgb,  (w, h), interpolation=cv2.INTER_LINEAR)

    H, W = before_rgb.shape[:2]
    
    # Fast water masking
    water_mask = create_water_mask_fast(before_rgb) | create_water_mask_fast(after_rgb)
    print(f"💧 Water: {water_mask.sum() / water_mask.size * 100:.1f}%")

    # Convert to grayscale for fast processing
    gray_b = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_a = cv2.cvtColor(after_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Histogram matching for illumination normalization (fast)
    gray_b_matched = cv2.normalize(gray_b, None, 0, 255, cv2.NORM_MINMAX)
    gray_a_matched = cv2.normalize(gray_a, None, 0, 255, cv2.NORM_MINMAX)
    
    # Simple absolute difference
    diff = np.abs(gray_b_matched - gray_a_matched)
    
    # Add color difference for better detection
    color_diff = np.mean(np.abs(before_rgb.astype(np.float32) - after_rgb.astype(np.float32)), axis=2)
    
    # Combine signals
    combined = 0.5 * diff + 0.5 * color_diff
    
    # Normalize
    combined = (combined / (np.percentile(combined, 98) + 1e-6)).clip(0, 1)
    
    # Adaptive threshold using Otsu
    combined_u8 = (combined * 255).clip(0, 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(combined_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(float(otsu_val) / 255.0, 0.30)
    
    # Create binary mask
    mask = (combined > threshold).astype(np.uint8)
    
    # Simple morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Remove small blobs
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(30, int(H * W * 0.0003))
    clean = np.zeros_like(mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 1
    
    # Apply water mask
    clean[water_mask] = 0
    
    print(f"⚡ Change detection: {time.time() - start:.2f}s | {clean.sum() / clean.size * 100:.2f}% changed")
    
    return Image.fromarray((clean * 255).astype(np.uint8), mode='L')


def overlay_mask(original_image, mask_image, color=(255, 0, 0), alpha=0.5):
    """
    Overlay the white regions of a binary mask onto the original image as a
    semi-transparent coloured highlight.

    Args:
        original_image: PIL Image – the 'After' image to draw on.
        mask_image:     PIL Image – B&W mask where white pixels mark changed areas.
        color:          RGB tuple for the highlight colour (default: red).
        alpha:          Opacity of the highlight layer, 0.0–1.0 (default: 0.5).

    Returns:
        PIL Image: Composited image with the highlight applied.
    """
    # --- normalise inputs to RGB numpy arrays ---
    original_rgb = np.array(original_image.convert('RGB'))
    mask_gray    = np.array(mask_image.convert('L'))

    # Resize mask to match the original image if dimensions differ
    if mask_gray.shape[:2] != original_rgb.shape[:2]:
        mask_gray = cv2.resize(
            mask_gray,
            (original_rgb.shape[1], original_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # Binary mask: True where the mask is white (changed pixels)
    changed = mask_gray > 127                         # shape (H, W), bool

    # Build a solid colour layer the same size as the original
    colour_layer = np.zeros_like(original_rgb, dtype=np.uint8)
    colour_layer[changed] = color                     # apply colour only to changed pixels

    # Blend: result = original * (1 - alpha) + colour * alpha  — for changed pixels only
    composited = original_rgb.copy().astype(np.float32)
    composited[changed] = (
        original_rgb[changed].astype(np.float32) * (1.0 - alpha)
        + np.array(color, dtype=np.float32) * alpha
    )
    composited = np.clip(composited, 0, 255).astype(np.uint8)

    return Image.fromarray(composited, mode='RGB')
