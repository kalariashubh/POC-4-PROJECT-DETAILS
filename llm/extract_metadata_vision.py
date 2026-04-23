import json
import base64
import re
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()

SYSTEM_PROMPT = """
You are a senior architectural & structural drawing analyst.

You are given an image of a DRAWING TITLE BLOCK.

Extract ONLY the following fields and return STRICT JSON:

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
- Use layout, not random text
- Project title is written near or under JOB / PROJECT TITLE
- Architect is near ARCHITECT label or logo
- Structural consultant is near its logo
- Never use legend / notes / beam text
- Never hallucinate
- Drawing number may be written as DRG NO / DWG NO / DRAWING NO
- Use null if uncertain
- Return ONLY JSON
"""

def _is_effectively_blank(text: str) -> bool:
    """
    Detects if the vision model indicates no readable content.
    """
    if not text:
        return True

    t = text.lower()
    blank_indicators = [
        "no text visible",
        "blank",
        "cannot see",
        "nothing readable",
        "image does not contain",
        "unclear",
        "no title block",
        "unable to identify"
    ]
    return any(k in t for k in blank_indicators)

def _clean_architect(value):
    if not value:
        return None

    v = value.upper()
    if re.search(r"(CLEAR HEIGHT|MM|SSL|BEAM|COLUMN|NC\d+)", v):
        return None

    if len(v) < 6 or len(v) > 80:
        return None

    return value.title()

def _extract_json_safe(text: str) -> dict:
    """
    Extract first valid JSON object from model output.
    """
    if not text:
        raise ValueError("Empty model response")

    text = text.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON found in model output:\n{text}")

    return json.loads(match.group())

def extract_metadata_from_image(image_path):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": SYSTEM_PROMPT},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{img_b64}",
                    },
                ],
            }
        ],
    )

    raw = response.output_text

    if _is_effectively_blank(raw):
        return {
            "project_title": None,
            "drawing_title": None,
            "architect": None,
            "structural_consultant": None,
            "drawing_no": None,
            "date": None,
            "email": None
        }

    try:
        data = _extract_json_safe(raw)
    except Exception:
        print("❌ MODEL OUTPUT WAS:")
        print(raw)
        raise RuntimeError("Vision model did not return valid JSON")

    data["architect"] = _clean_architect(data.get("architect"))

    return data
