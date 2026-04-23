# POC-4: Project Details Extraction (Vision + OCR Pipeline)

## Overview

This project extracts structured **project/title block details** from engineering drawings using a hybrid pipeline combining **OCR + Vision-based AI**.

The system automatically:
- Detects relevant keywords using OCR
- Crops the title block region
- Extracts structured metadata using a vision model
- Saves results in JSON format along with the cropped image

---

## 🔄 Pipeline Flow

run_pipeline.py  
 ├── run_ocr.py                  → Detect keywords / regions  
 ├── crop_titleblock.py         → Crop title block area  
 ├── extract_metadata_vision.py → Extract structured data  
 └── Save outputs (JSON + image)

---

## 📁 Project Structure

inputs/                     → Input drawings (PDF/Image)  
outputs/                    → Generated results  

llm/  
 └── extract_metadata_vision.py  

ocr/  
 └── run_ocr.py  

vision/  
 └── crop_titleblock.py  

pipeline/  
 └── run_pipeline.py  

requirements.txt  
.env  
.gitignore  

---


---

## ⚙️ Extracted Output Format

```json
{
  "project_title": null,
  "drawing_title": null,
  "architect": null,
  "structural_consultant": null,
  "drawing_no": null,
  "date": null,
  "email": null
}

---

## 🚀 Usage

### 1. Install dependencies

pip install -r requirements.txt

### 2. Set environment variables

Create a `.env` file:

OPENAI_API_KEY=your_api_key_here

### 3. Run the pipeline

python pipeline/run_pipeline.py

---

## 🧠 Notes

- Works best with clean engineering drawings  
- Requires Tesseract OCR installed and added to PATH  
- Supports PDF and image inputs  
- Modular design allows easy extension  

---

## 🔧 Tech Stack

- Python  
- OpenCV  
- Tesseract OCR  
- OpenAI Vision Model  
- pdf2image  
