import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw
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
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from functools import partial
from torch import nn as nn
def build_ssdlite(num_classes=2):
    model = ssdlite320_mobilenet_v3_large(weights='DEFAULT')

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

                loaded = torch.load(path, map_location=self.device)
                print(f"\n📦 DEBUG {model_file}")
                print(type(loaded))

                if isinstance(loaded, dict):
                    print("keys:", loaded.keys())

                # ---------------- FRCNN ----------------
                if model_file == "frcnn_v2.pt":
                    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)

                    in_features = model.roi_heads.box_predictor.cls_score.in_features
                    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

                    model.load_state_dict(loaded, strict=False)
                    print("✅ FRCNN загружена")

                elif model_file == "ssd_v2.pt":
                    model = build_ssdlite(num_classes=2)

                    model.load_state_dict(loaded)

                    print("✅ SSD загружена ПРАВИЛЬНО (через build_ssdlite)")

                # ---------------- CRNN ----------------
                elif model_file == "crnn.pt":
                    model = {
                        "type": "state_dict",
                        "state_dict": loaded,
                        "name": "crnn"
                    }
                    print("⚠️ CRNN как state_dict")

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

        print(f"\n🎉 Всего моделей: {list(self.models.keys())}")

    # =========================================================

    def predict(self, image_bytes: bytes, model_name: str):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original = self._to_base64(image)

        if model_name not in self.models:
            return {"error": "Model not found"}, original, original

        model = self.models[model_name]

        try:
            # ---------------- YOLO ----------------
            if "yolo" in model_name.lower():
                results = model(image, conf=0.25)[0]
                annotated = Image.fromarray(results.plot())

                return {
                    "type": "object_detection",
                    "boxes": results.boxes.xyxy.tolist() if results.boxes else []
                }, original, self._to_base64(annotated)

            # ---------------- FRCNN ----------------
            elif model_name == "frcnn_v2.pt":
                return self._run_torchvision_model(image, model, color="red", thresh=0.6)

            # ---------------- SSD ----------------
            elif model_name == "ssd_v2.pt":
                return self._run_torchvision_model(image, model, color="blue", thresh=0.5)

            # ---------------- CRNN ----------------
            elif model_name == "crnn.pt":
                return {
                    "type": "ocr",
                    "message": "CRNN пока только state_dict"
                }, original, original

            return {"error": "Unknown model"}, original, original

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return {"error": str(e)}, original, original

    # =========================================================

    def _run_torchvision_model(self, image, model, color="red", thresh=0.05):

        if isinstance(model, dict):
            return {
                "type": "error",
                "message": "Model is dict, not loaded properly"
            }, self._to_base64(image), self._to_base64(image)

        transform = T.Compose([T.ToTensor()])
        tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = model(tensor)[0]

        print("\n--- DEBUG ---")
        print("scores max:", output["scores"].max().item())
        print("scores min:", output["scores"].min().item())

        boxes = output["boxes"]
        scores = output["scores"]
        labels = output.get("labels", None)

        # 🔥 LOWER THRESH
        keep = scores > thresh
        boxes = boxes[keep]
        scores = scores[keep]

        if labels is not None:
            labels = labels[keep]

        if len(scores) == 0:
            return {
                "type": "object_detection",
                "boxes": [],
                "scores": []
            }, self._to_base64(image), self._to_base64(image)

        keep_idx = ops.nms(boxes, scores, 0.4)
        boxes = boxes[keep_idx][:20]
        scores = scores[keep_idx][:20]

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        for i, (box, score) in enumerate(zip(boxes, scores)):
            b = box.cpu().numpy()
            draw.rectangle(b.tolist(), outline=color, width=3)
            draw.text((b[0], b[1] - 10), f"{score:.2f}", fill=color)

        return {
            "type": "object_detection",
            "boxes": boxes.cpu().numpy().tolist(),
            "scores": scores.cpu().numpy().tolist()
        }, self._to_base64(image), self._to_base64(annotated)

    # =========================================================

    def _to_base64(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()