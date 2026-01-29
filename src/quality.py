import cv2
import numpy as np

def quality_checks(img_bgr: np.ndarray) -> dict:
    """
    Returns a dict with:
    - passed: bool
    - warnings: list[str]
    - stats: dict
    """
    warnings = []

    if img_bgr is None or img_bgr.size == 0:
        return {"passed": False, "warnings": ["Invalid image"], "stats": {}}

    h, w = img_bgr.shape[:2]
    if min(h, w) < 224:
        warnings.append("Low resolution image (recommended >= 224px).")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Brightness check
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 40:
        warnings.append("Image is very dark (low brightness).")
    if mean_brightness > 215:
        warnings.append("Image is very bright (high brightness).")

    # Blur check (Laplacian variance)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < 80:
        warnings.append("Image may be blurry (low sharpness).")

    # Contrast check (std dev)
    contrast = float(np.std(gray))
    if contrast < 25:
        warnings.append("Low contrast image (details may be unclear).")

    # Simple pass rule: allow warnings but block only extreme issues
    hard_block = (
        (min(h, w) < 160) or
        (mean_brightness < 20) or
        (mean_brightness > 240)
    )

    passed = not hard_block
    if hard_block:
        warnings.insert(0, "Quality gate failed: image may be unreliable for prediction.")

    return {
        "passed": passed,
        "warnings": warnings,
        "stats": {
            "height": h,
            "width": w,
            "mean_brightness": mean_brightness,
            "blur_score": blur_score,
            "contrast": contrast
        }
    }
