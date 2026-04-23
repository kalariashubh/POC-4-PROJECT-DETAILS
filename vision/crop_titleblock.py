import cv2
from pathlib import Path
from typing import List, Tuple

KEYWORDS = [
    "PROJECT",
    "JOB",
    "DRAWING",
    "STRUCTURAL",
    "CONSULTANT",
    "DRG",
    "DATE",
    "SCALE",
    "SHEET",
    "ARCHITECT",
]

MIN_KEYWORD_SCORE = 2

def _score_text(text: str) -> int:
    text = text.upper()
    return sum(1 for k in KEYWORDS if k in text)

def generate_candidate_crops(img):
    h, w, _ = img.shape

    candidates = {
        "right_strip": img[:, int(w * 0.80):w],
        "bottom_strip": img[int(h * 0.80):h, :],
        "bottom_right": img[int(h * 0.45):h, int(w * 0.45):w],
        "top_right": img[0:int(h * 0.35), int(w * 0.60):w],
    }
    return candidates

def crop_titleblock(page_img: Path, output_img: Path, ocr_fn):
    """
    ocr_fn = function that takes image ndarray and returns text
    """
    img = cv2.imread(str(page_img))
    candidates = generate_candidate_crops(img)

    best_score = 0
    best_crop = None

    for name, crop in candidates.items():
        text = ocr_fn(crop)
        score = _score_text(text)

        print(f"[CROP TEST] {name} → score={score}")

        if score > best_score:
            best_score = score
            best_crop = crop

    if best_score < MIN_KEYWORD_SCORE:
        print("[WARN] No title block keywords detected — skipping crop")
        return None

    cv2.imwrite(str(output_img), best_crop)
    return output_img
