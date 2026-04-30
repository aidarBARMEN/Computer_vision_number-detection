import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance, ImageFilter
import io
import base64
import os
import numpy as np
import torchvision.transforms as T
import torchvision.ops as ops

from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    ssdlite320_mobilenet_v3_large
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from functools import partial
from torch import nn as nn

from ocr import CRNNRecognizer, EasyOCRRecognizer


def build_ssdlite(num_classes=2):
    # weights=None: the user's ssd_v2.pt state_dict supplies everything;
    # avoids a network fetch of the COCO-pretrained checkpoint.
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)

    in_channels = [
        model.head.classification_head.module_list[i][0][0].in_channels
        for i in range(len(model.head.classification_head.module_list))
    ]

    num_anchors = model.anchor_generator.num_anchors_per_location()

    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    model.head.classification_head = SSDLiteClassificationHead(
        in_channels,
        num_anchors,
        num_classes,
        norm_layer
    )

    return model


class CVModelManager:
    def __init__(self):
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = "models"

        print(f"🚀 Device: {self.device}")

        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pt')]
        print(f"📦 Найдено моделей: {model_files}")

        for model_file in model_files:
            path = os.path.join(self.model_dir, model_file)

            try:
                # ---------------- YOLO ----------------
                if 'yolo' in model_file.lower():
                    self.models[model_file] = YOLO(path)
                    print(f"✅ YOLO: {model_file}")
                    continue

                # CRNN handled separately by CRNNRecognizer
                if model_file == "crnn.pt":
                    continue

                loaded = torch.load(path, map_location=self.device, weights_only=False)
                print(f"\n📦 DEBUG {model_file}")
                print(type(loaded))

                # ---------------- FRCNN ----------------
                if model_file == "frcnn_v2.pt":
                    model = fasterrcnn_mobilenet_v3_large_320_fpn(
                        weights=None, weights_backbone=None
                    )

                    in_features = model.roi_heads.box_predictor.cls_score.in_features
                    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

                    model.load_state_dict(loaded, strict=False)
                    print("✅ FRCNN загружена")

                elif model_file == "ssd_v2.pt":
                    model = build_ssdlite(num_classes=2)
                    model.load_state_dict(loaded)
                    print("✅ SSD загружена")

                else:
                    model = loaded

                if hasattr(model, "eval"):
                    model.eval()
                if hasattr(model, "to"):
                    model = model.to(self.device)

                self.models[model_file] = model
                print(f"✅ Добавлена: {model_file}")

            except Exception as e:
                import traceback
                print(f"\n💥 FULL ERROR {model_file}")
                print(traceback.format_exc())

        # ---------------- OCR ----------------
        crnn_path = os.path.join(self.model_dir, "crnn.pt")
        try:
            self.crnn = CRNNRecognizer(crnn_path, self.device)
            print("✅ CRNN OCR (custom) готов")
        except Exception as e:
            import traceback
            print("💥 CRNN init failed:", traceback.format_exc())
            self.crnn = None

        self.easyocr = EasyOCRRecognizer(self.device)
        print("✅ EasyOCR (existing) зарегистрирован (lazy init)")

        print(f"\n🎉 Всего моделей: {list(self.models.keys())} + crnn.pt + easyocr")

    # =========================================================

    def predict(self, image_bytes: bytes, model_name: str, ocr_engine: str = "both"):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original = self._to_base64(image)

        try:
            # ---------------- Pure CRNN OCR ----------------
            if model_name == "crnn.pt":
                if self.crnn is None:
                    return {"error": "CRNN not loaded"}, original, original
                text = self.crnn.recognize(image)
                annotated = self._annotate_text_only(image, text, label="CRNN")
                return {
                    "type": "ocr",
                    "engine": "crnn (custom)",
                    "text": text,
                }, original, self._to_base64(annotated)

            # ---------------- Pure EasyOCR ----------------
            if model_name == "easyocr":
                text = self.easyocr.recognize(image)
                annotated = self._annotate_text_only(image, text, label="EasyOCR")
                return {
                    "type": "ocr",
                    "engine": "easyocr (existing)",
                    "text": text,
                }, original, self._to_base64(annotated)

            if model_name not in self.models:
                return {"error": "Model not found"}, original, original

            model = self.models[model_name]

            # ---------------- Detection + OCR ----------------
            stage_info = None
            if "yolo" in model_name.lower():
                # COCO YOLO finds vehicles, not plates → cascade to plate detector.
                # Classes 2/3/5/7 = car/motorcycle/bus/truck.
                results = model(image, conf=0.25, classes=[2, 3, 5, 7])[0]
                vehicle_boxes = []
                if results.boxes is not None and len(results.boxes) > 0:
                    vehicle_boxes = results.boxes.xyxy.cpu().numpy().tolist()

                plate_detector = self.models.get("frcnn_v2.pt")
                if plate_detector is None:
                    return {"error": "Plate detector frcnn_v2.pt missing"}, original, original

                if vehicle_boxes:
                    boxes = self._cascade_plates(image, vehicle_boxes, plate_detector)
                    stage_info = f"YOLO нашёл {len(vehicle_boxes)} ТС → FRCNN нашёл {len(boxes)} номер(ов)"
                else:
                    # YOLO found no vehicle → fall back to running FRCNN on full image.
                    boxes = self._run_torchvision_boxes(image, plate_detector, thresh=0.6)
                    stage_info = f"YOLO ничего не нашёл, fallback FRCNN: {len(boxes)} номер(ов)"

            elif model_name == "frcnn_v2.pt":
                boxes = self._run_torchvision_boxes(image, model, thresh=0.6)

            elif model_name == "ssd_v2.pt":
                boxes = self._run_torchvision_boxes(image, model, thresh=0.5)

            else:
                return {"error": "Unknown model"}, original, original

            recognitions = self._recognize_boxes(image, boxes, ocr_engine)
            annotated = self._draw_results(image, boxes, recognitions, model_name)

            payload = {
                "type": "detection_ocr",
                "model": model_name,
                "ocr_engine": ocr_engine,
                "boxes": boxes,
                "recognitions": recognitions,
            }
            if stage_info is not None:
                payload["pipeline"] = stage_info
            return payload, original, self._to_base64(annotated)

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return {"error": str(e)}, original, original

    # =========================================================

    def _run_torchvision_boxes(self, image, model, thresh=0.5):
        transform = T.Compose([T.ToTensor()])
        tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = model(tensor)[0]

        boxes = output["boxes"]
        scores = output["scores"]

        keep = scores > thresh
        boxes = boxes[keep]
        scores = scores[keep]

        if len(scores) == 0:
            return []

        keep_idx = ops.nms(boxes, scores, 0.4)
        boxes = boxes[keep_idx][:20]
        return boxes.cpu().numpy().tolist()

    def _cascade_plates(self, image: Image.Image, vehicle_boxes, plate_detector, thresh=0.5):
        """For each vehicle bbox, run plate detector and return plate boxes in image coords."""
        all_plates = []
        W, H = image.size
        for vbox in vehicle_boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in vbox]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W, x2); y2 = min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image.crop((x1, y1, x2, y2))
            local_boxes = self._run_torchvision_boxes(crop, plate_detector, thresh=thresh)
            for lb in local_boxes:
                lx1, ly1, lx2, ly2 = lb
                all_plates.append([lx1 + x1, ly1 + y1, lx2 + x1, ly2 + y1])
        return all_plates

    def _recognize_boxes(self, image: Image.Image, boxes, ocr_engine: str):
        recognitions = []
        W, H = image.size
        for box in boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            # Expand box by ~8% on each side to capture plate borders/letters cropped
            # at the edge. CRNN/EasyOCR both benefit from a small margin.
            bw, bh = x2 - x1, y2 - y1
            pad_x = max(2, int(bw * 0.08))
            pad_y = max(2, int(bh * 0.12))
            ex1 = max(0, min(x1 - pad_x, W - 1))
            ey1 = max(0, min(y1 - pad_y, H - 1))
            ex2 = max(ex1 + 1, min(x2 + pad_x, W))
            ey2 = max(ey1 + 1, min(y2 + pad_y, H))
            crop = image.crop((ex1, ey1, ex2, ey2))
            crop_proc = self._enhance_for_ocr(crop)

            entry = {"box": [x1, y1, x2, y2]}
            if ocr_engine in ("crnn", "both") and self.crnn is not None:
                try:
                    entry["crnn"] = self.crnn.recognize(crop_proc)
                except Exception as e:
                    entry["crnn_error"] = str(e)
            if ocr_engine in ("easyocr", "both"):
                try:
                    entry["easyocr"] = self.easyocr.recognize(crop)
                except Exception as e:
                    entry["easyocr_error"] = str(e)
            recognitions.append(entry)
        return recognitions

    @staticmethod
    def _enhance_for_ocr(crop: Image.Image) -> Image.Image:
        """Upscale + autocontrast + sharpen for tiny plate crops."""
        w, h = crop.size
        if h < 48:
            scale = 48 / max(h, 1)
            crop = crop.resize((max(1, int(w * scale)), 48), Image.BICUBIC)
        crop = ImageOps.autocontrast(crop, cutoff=2)
        crop = ImageEnhance.Contrast(crop).enhance(1.4)
        crop = crop.filter(ImageFilter.SHARPEN)
        return crop

    def _draw_results(self, image: Image.Image, boxes, recognitions, model_name: str):
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial.ttf", 22
            )
        except Exception:
            font = ImageFont.load_default()

        color = "lime" if "yolo" in model_name.lower() else (
            "red" if model_name == "frcnn_v2.pt" else "deepskyblue"
        )

        for box, rec in zip(boxes, recognitions):
            x1, y1, x2, y2 = rec["box"]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            parts = []
            if "crnn" in rec and rec["crnn"]:
                parts.append(f"CRNN: {rec['crnn']}")
            if "easyocr" in rec and rec["easyocr"]:
                parts.append(f"OCR: {rec['easyocr']}")
            label = " | ".join(parts) if parts else "?"

            tw, th = self._text_size(draw, label, font)
            ty = max(0, y1 - th - 6)
            draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=color)
            draw.text((x1 + 4, ty + 2), label, fill="black", font=font)

        return annotated

    def _annotate_text_only(self, image: Image.Image, text: str, label: str):
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial.ttf", 28
            )
        except Exception:
            font = ImageFont.load_default()
        msg = f"{label}: {text or '?'}"
        tw, th = self._text_size(draw, msg, font)
        draw.rectangle([5, 5, 5 + tw + 12, 5 + th + 10], fill="yellow")
        draw.text((11, 10), msg, fill="black", font=font)
        return annotated

    @staticmethod
    def _text_size(draw, text, font):
        try:
            l, t, r, b = draw.textbbox((0, 0), text, font=font)
            return r - l, b - t
        except Exception:
            return font.getsize(text) if hasattr(font, "getsize") else (len(text) * 8, 16)

    # =========================================================

    def _to_base64(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
