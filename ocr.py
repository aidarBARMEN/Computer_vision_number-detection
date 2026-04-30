import torch
import torch.nn as nn
import numpy as np
from PIL import Image


CRNN_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


class CRNN(nn.Module):
    def __init__(self, num_classes: int = 37, hidden_size: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),       # 0
            nn.ReLU(inplace=True),            # 1
            nn.MaxPool2d(2, 2),               # 2  -> 16
            nn.Conv2d(64, 128, 3, 1, 1),      # 3
            nn.ReLU(inplace=True),            # 4
            nn.MaxPool2d(2, 2),               # 5  -> 8
            nn.Conv2d(128, 256, 3, 1, 1),     # 6
            nn.BatchNorm2d(256),              # 7
            nn.ReLU(inplace=True),            # 8
            nn.Conv2d(256, 256, 3, 1, 1),     # 9
            nn.ReLU(inplace=True),            # 10
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 11 -> 4
            nn.Conv2d(256, 512, 3, 1, 1),     # 12
            nn.BatchNorm2d(512),              # 13
            nn.ReLU(inplace=True),            # 14
            nn.Conv2d(512, 512, 3, 1, 1),     # 15
            nn.ReLU(inplace=True),            # 16
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 17 -> 2
            nn.Conv2d(512, 512, 2, 1, 0),     # 18 -> 1
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x):
        feat = self.cnn(x)                      # (B, 512, 1, T)
        b, c, h, w = feat.shape
        assert h == 1, f"expected feature height 1, got {h}"
        feat = feat.squeeze(2).permute(2, 0, 1)  # (T, B, 512)
        rnn_out, _ = self.rnn(feat)              # (T, B, 2*hidden)
        # Match the saved fc.weight shape (37, 512): sum the two directions
        t, bb, hh = rnn_out.shape
        rnn_out = rnn_out.view(t, bb, 2, hh // 2).sum(dim=2)  # (T, B, hidden) — but fc expects 512
        return self.fc(rnn_out)


def _build_crnn_matching_state_dict(state_dict, device):
    """Build a CRNN whose Linear matches saved fc weights (37, 512)."""
    fc_w = state_dict["fc.weight"]
    out_features, in_features = fc_w.shape

    class _CRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
                nn.Conv2d(256, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
                nn.Conv2d(512, 512, 2, 1, 0),
                nn.ReLU(inplace=True),
            )
            self.rnn = nn.LSTM(
                input_size=512, hidden_size=256, num_layers=2, bidirectional=True
            )
            self.fc = nn.Linear(in_features, out_features)

        def forward(self, x):
            feat = self.cnn(x)
            b, c, h, w = feat.shape
            feat = feat.squeeze(2).permute(2, 0, 1)
            rnn_out, _ = self.rnn(feat)
            # rnn_out: (T, B, 512) when bidirectional with hidden=256
            return self.fc(rnn_out)

    model = _CRNN()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model


def _ctc_greedy_decode(logits: torch.Tensor, alphabet: str = CRNN_ALPHABET) -> str:
    """logits: (T, num_classes). Last class index is the CTC blank."""
    blank = len(alphabet)  # 36 for a 37-class model
    pred = logits.argmax(dim=-1).tolist()
    out = []
    prev = -1
    for p in pred:
        if p != prev and p != blank:
            if 0 <= p < len(alphabet):
                out.append(alphabet[p])
        prev = p
    return "".join(out)


class CRNNRecognizer:
    """Custom CRNN OCR using the project's crnn.pt weights."""

    def __init__(self, weights_path: str, device: torch.device):
        self.device = device
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        self.model = _build_crnn_matching_state_dict(state_dict, device)
        self.target_h = 32
        self.target_w = 100

    def _preprocess(self, pil_img: Image.Image) -> torch.Tensor:
        from PIL import ImageOps, ImageEnhance, ImageFilter
        gray = pil_img.convert("L")
        # Boost contrast & sharpen — tiny plate crops are usually low-contrast.
        gray = ImageOps.autocontrast(gray, cutoff=2)
        gray = ImageEnhance.Contrast(gray).enhance(1.3)
        gray = gray.filter(ImageFilter.SHARPEN)
        w, h = gray.size
        if h == 0 or w == 0:
            gray = Image.new("L", (self.target_w, self.target_h), color=0)
        else:
            new_w = max(1, int(w * (self.target_h / h)))
            # Wider window — Russian plates are long; clamp to reasonable bounds.
            new_w = min(max(new_w, 64), 512)
            gray = gray.resize((new_w, self.target_h), Image.BICUBIC)
        arr = np.asarray(gray, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return tensor.to(self.device)

    @torch.no_grad()
    def recognize(self, pil_img: Image.Image) -> str:
        x = self._preprocess(pil_img)
        logits = self.model(x)            # (T, 1, num_classes)
        logits = logits.squeeze(1).cpu()  # (T, num_classes)
        text = _ctc_greedy_decode(logits)
        return text.upper()


PLATE_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class EasyOCRRecognizer:
    """Existing solution: EasyOCR. Lazily initialized."""

    def __init__(self, device: torch.device):
        self._reader = None
        self._device = device

    def _ensure(self):
        if self._reader is None:
            import easyocr
            gpu = self._device.type == "cuda"
            self._reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        return self._reader

    def recognize(self, pil_img: Image.Image) -> str:
        reader = self._ensure()
        arr = np.array(pil_img.convert("RGB"))
        # Restrict alphabet to plate-friendly chars and concatenate all detected
        # fragments (helps when EasyOCR splits a plate into 2-3 pieces).
        results = reader.readtext(
            arr,
            detail=1,
            paragraph=False,
            allowlist=PLATE_ALLOWLIST,
            text_threshold=0.4,
            low_text=0.3,
        )
        if not results:
            return ""
        # Sort left-to-right by bbox x, drop fragments shorter than 2 chars.
        fragments = []
        for bbox, txt, conf in results:
            cleaned = "".join(ch for ch in txt.upper() if ch.isalnum())
            if len(cleaned) < 2 or conf < 0.2:
                continue
            x = min(p[0] for p in bbox)
            fragments.append((x, cleaned, conf))
        if not fragments:
            best = max(results, key=lambda r: r[2])
            return "".join(c for c in best[1].upper() if c.isalnum())
        fragments.sort(key=lambda t: t[0])
        return "".join(f[1] for f in fragments)
