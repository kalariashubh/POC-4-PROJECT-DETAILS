from pathlib import Path
from pdf2image import convert_from_path
from vision.crop_titleblock import crop_titleblock
from llm.extract_metadata_vision import extract_metadata_from_image
from ocr.run_ocr import run_ocr_image
import json

POPPLER_PATH = r"C:\poppler\Library\bin"

INPUT_PDF = Path("inputs/test1.pdf")

OUT_ROOT = Path("outputs")
OUT_ROOT.mkdir(exist_ok=True)

pdf_name = INPUT_PDF.stem               
OUT = OUT_ROOT / pdf_name             
OUT.mkdir(exist_ok=True)

print("[1] Convert PDF → image")

pages = convert_from_path(
    INPUT_PDF,
    dpi=200,
    first_page=1,
    last_page=1,
    poppler_path=POPPLER_PATH
)

page_img = OUT / f"{pdf_name}.png"
pages[0].save(page_img)

print("[2] Crop title block")

crop_img = OUT / f"{pdf_name}_crop.png"
crop_result = crop_titleblock(page_img, crop_img, run_ocr_image)

print("[3] GPT-Vision extraction")

if crop_result is None:
    print("[INFO] No title block detected — returning null metadata")

    metadata = {
        "project_title": None,
        "drawing_title": None,
        "architect": None,
        "structural_consultant": None,
        "drawing_no": None,
        "date": None,
        "email": None
    }
else:
    metadata = extract_metadata_from_image(crop_img)

json_path = OUT / f"{pdf_name}.json"
json_path.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

print("[OK] DONE")
print(json.dumps(metadata, indent=2))
