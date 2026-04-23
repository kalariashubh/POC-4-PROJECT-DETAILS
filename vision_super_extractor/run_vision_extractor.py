import os
import json
import base64
import cv2
import numpy as np

from pathlib import Path
from dotenv import load_dotenv
from pdf2image import convert_from_path
from openai import OpenAI


# ================================
# LOAD ENV
# ================================

load_dotenv()

client = OpenAI()


# ================================
# CONFIG
# ================================

POPPLER_PATH = r"C:\poppler\Library\bin"   # CHANGE if needed
INPUT_PDF = Path("0001_O25060-C-HP-02-NU-0001-R0-SETTINGOUT DETAIL OF COLUMN &amp; SHEAR WALL(SHEET 1 OF 3)_rotated.pdf")              # <-- just drop your PDF here

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ================================
# STEP 1 — PDF → HIGH DPI IMAGE
# ================================

print("\n[1] Converting PDF to high-resolution image...")

pages = convert_from_path(
    INPUT_PDF,
    dpi=300,            # 🔥 CRITICAL FOR SCANNED DRAWINGS
    poppler_path=POPPLER_PATH
)

page_img = OUTPUT_DIR / "page.png"
pages[0].save(page_img)

print("✅ Image saved.")


# ================================
# STEP 2 — AUTO ROTATE
# ================================

print("\n[2] Detecting orientation...")

img = cv2.imread(str(page_img))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)

coords = np.column_stack(np.where(gray > 0))
angle = cv2.minAreaRect(coords)[-1]

if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle

(h, w) = img.shape[:2]
center = (w // 2, h // 2)

M = cv2.getRotationMatrix2D(center, angle, 1.0)

rotated = cv2.warpAffine(
    img,
    M,
    (w, h),
    flags=cv2.INTER_CUBIC,
    borderMode=cv2.BORDER_REPLICATE,
)

rotated_path = OUTPUT_DIR / "rotated.png"
cv2.imwrite(str(rotated_path), rotated)

print(f"✅ Rotation corrected ({angle:.2f}°).")


# ================================
# STEP 3 — ENHANCE SCAN
# ================================

print("\n[3] Enhancing blueprint readability...")

gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

# Increase contrast
gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=25)

# Sharpen
kernel = np.array([
    [0, -1, 0],
    [-1, 5,-1],
    [0, -1, 0]
])

enhanced = cv2.filter2D(gray, -1, kernel)

enhanced_path = OUTPUT_DIR / "enhanced.png"
cv2.imwrite(str(enhanced_path), enhanced)

print("✅ Enhancement complete.")


# ================================
# STEP 4 — SEND TO GPT VISION
# ================================

print("\n[4] Sending image to GPT Vision...")

with open(enhanced_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()


SYSTEM_PROMPT = """
You are a senior structural drawing analyst.

This is a construction drawing.

FIRST visually locate the TITLE BLOCK.
Then extract ONLY the fields below.

Return STRICT JSON:

{
  "project_title": null,
  "drawing_title": null,
  "architect": null,
  "structural_consultant": null,
  "drawing_no": null,
  "date": null,
  "email": null
}

Rules:

- Ignore legends and notes
- Ignore beam schedules
- Ignore random annotations
- Focus ONLY on title block
- Never hallucinate
- If unreadable → return null
- Output JSON only
"""


response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": SYSTEM_PROMPT},
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{img_b64}"
            },
        ],
    }],
)

raw = response.output_text.strip()


# ================================
# SAFE JSON EXTRACTION
# ================================

import re

match = re.search(r"\{[\s\S]*\}", raw)

if not match:
    print("\n❌ MODEL OUTPUT:")
    print(raw)
    raise RuntimeError("Vision model did not return JSON.")

data = json.loads(match.group())


# ================================
# STEP 5 — SAVE JSON
# ================================

json_path = OUTPUT_DIR / "metadata.json"

json_path.write_text(
    json.dumps(data, indent=2),
    encoding="utf-8"
)

print("\n✅ EXTRACTION COMPLETE!\n")
print(json.dumps(data, indent=2))
