# 🚗 License Plate Recognition — CV FastAPI

**FastAPI** computer-vision service for **car license plate detection + OCR**.

Combines **5 detection / OCR models** into one cascaded pipeline:
**YOLO → Faster R-CNN → CRNN / EasyOCR**.

---

## ✨ Features

* Cascaded detection pipeline:
  1. **YOLOv8n / YOLO26n** — find vehicles (cars / trucks / buses / motorcycles, COCO classes 2/3/5/7)
  2. **Faster R-CNN v2** — find license plates *inside* each vehicle crop
  3. **CRNN (custom)** + **EasyOCR (baseline)** — read plate text
* Automatic fallback: if YOLO finds no vehicles → FRCNN runs on the full image
* Two OCR engines side-by-side: own solution (CRNN + CTC) vs library baseline (EasyOCR with plate allowlist)
* OCR enhancement: box expansion (+8 % / +12 %), autocontrast, sharpening, upscaling for tiny crops
* Modern web UI (Bootstrap 5 + JetBrains Mono) with yellow license-plate banner and copy-JSON
* Runs on CPU, auto-uses GPU if available

### Model status

| Model            | Role                       | Status              |
| :--------------- | :------------------------- | :------------------ |
| `yolov8n.pt`     | Vehicle detector           | ✅ Working           |
| `yolo26n.pt`     | Vehicle detector           | ✅ Working           |
| `frcnn_v2.pt`    | Plate detector             | ✅ Working           |
| `ssd_v2.pt`      | Plate detector (alt.)      | ⚠️ State-dict loads |
| `crnn.pt`        | Custom OCR (CNN+BiLSTM+CTC)| ✅ Working           |
| `easyocr`        | OCR baseline (library)     | ✅ Working           |

---

## 🚀 How to Run

### 1. Go to project folder

```bash
cd Computer_vision_number-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place model files

Put all 5 weight files in the `models/` folder:

```
models/
├── crnn.pt
├── frcnn_v2.pt
├── ssd_v2.pt
├── yolo26n.pt
└── yolov8n.pt
```

### 4. Start the server

```bash
python main.py
```

Open in browser: **http://127.0.0.1:8000**

---

## 🗂 Project Structure

```text
Computer_vision_number-detection/
├── models/             # all .pt weight files
├── templates/
│   └── index.html      # web UI (Bootstrap + custom plate theme)
├── main.py             # FastAPI app — / and /predict endpoints
├── inference.py        # CVModelManager — model loading + cascade pipeline
├── ocr.py              # CRNNRecognizer + EasyOCRRecognizer
├── generate_report.py  # builds Project_Report.docx
├── requirements.txt
└── README.md
```

---

## 🧠 How the cascade pipeline works

When you pick a **YOLO** detector in the UI:

1. YOLO runs on the full image, filtered to vehicle classes only (COCO 2/3/5/7).
2. For each vehicle bbox, the image is cropped and **Faster R-CNN v2** runs on the crop to find license plates.
3. Plate boxes are mapped back into the original image coordinate space.
4. Each plate crop is expanded (~8 % / 12 %), enhanced (autocontrast + sharpen), and passed to the chosen OCR engine.
5. If YOLO finds no vehicles, FRCNN automatically runs on the **full image** as a fallback.

When you pick **Faster R-CNN v2** or **SSDLite v2** directly — they run plate detection on the full image without the YOLO stage.

When you pick **CRNN** or **EasyOCR** as the model — only OCR is performed (the whole image is treated as a single plate crop).

---

## 🖥 How to Use

1. Open **http://127.0.0.1:8000**
2. Pick a **detector** (YOLOv8n / YOLO26n / FRCNN / SSD / CRNN-only / EasyOCR-only)
3. Pick an **OCR engine** (`both` / `crnn` / `easyocr`)
4. Upload an image (JPG / PNG)
5. Click **🚀 Распознать номер**
6. You get back:
   * **Yellow plate banner** with the recognized number (CRNN + EasyOCR side-by-side when both selected)
   * **Original** and **annotated** images side-by-side
   * Full **JSON** response from `/predict` (with copy-to-clipboard button)

---

## 📡 API

### `POST /predict`

Form fields:

| Field         | Type   | Required | Notes                                                          |
| :------------ | :----- | :------- | :------------------------------------------------------------- |
| `file`        | file   | yes      | Image (JPG / PNG / WebP)                                       |
| `model_name`  | string | yes      | `yolov8n.pt` / `yolo26n.pt` / `frcnn_v2.pt` / `ssd_v2.pt` / `crnn.pt` / `easyocr` |
| `ocr_engine`  | string | no       | `both` (default) / `crnn` / `easyocr`                          |

Response (detection + OCR):

```json
{
  "prediction": {
    "type": "detection_ocr",
    "model": "yolov8n.pt",
    "ocr_engine": "both",
    "pipeline": "YOLO нашёл 1 ТС → FRCNN нашёл 1 номер(ов)",
    "boxes": [[412, 287, 698, 361]],
    "recognitions": [
      {"box": [412, 287, 698, 361], "crnn": "A123BC77", "easyocr": "A123BC77"}
    ]
  },
  "original_image": "data:image/jpeg;base64,...",
  "annotated_image": "data:image/jpeg;base64,...",
  "model_used": "yolov8n.pt",
  "ocr_engine": "both"
}
```

---

**Project:** CV FastAPI — License Plate Recognition
**Date:** April 30, 2026
