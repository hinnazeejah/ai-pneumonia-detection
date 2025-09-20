# app/inference.py
from pathlib import Path
from typing import Dict
import torch
from torchvision import transforms, models
from PIL import Image

# ---- Device selection (Apple MPS > CUDA > CPU) ----
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")        # Apple Silicon GPU
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")       # NVIDIA GPU
else:
    DEVICE = torch.device("cpu")        # Fallback CPU

class InferenceModel:
    """
    Loads a trained EfficientNet-B0 checkpoint and provides a predict() method.
    Expects a checkpoint saved by train.py with keys:
      - "state_dict": model state dict
      - "classes": list of class names (["NORMAL","PNEUMONIA"])
      - "img_size": int (e.g., 224)
    """

    def __init__(self, ckpt_path: str | Path):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        self.img_size: int = int(ckpt.get("img_size", 224))
        self.classes = list(ckpt["classes"])
        if len(self.classes) != 2 or not all(c in self.classes for c in ["NORMAL", "PNEUMONIA"]):
            raise ValueError(f"Expected classes ['NORMAL','PNEUMONIA'], got {self.classes}")

        # Build model and load weights
        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(in_features, len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(DEVICE)

        # Preprocessing (must match training normalization)
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        self.tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # cache indices for speed/clarity
        self.idx_normal = self.classes.index("NORMAL")
        self.idx_pneum  = self.classes.index("PNEUMONIA")

    @torch.inference_mode()
    def predict(self, pil_img: Image.Image, threshold: float = 0.85) -> Dict:
        """
        Predicts NORMAL vs PNEUMONIA with a probability threshold on PNEUMONIA.
        If P(PNEUMONIA) >= threshold -> label=PNEUMONIA else NORMAL.
        Returns: dict with 'label', 'confidence', and 'probs' per class.
        """
        if not isinstance(pil_img, Image.Image):
            raise TypeError("predict expects a PIL.Image.Image")

        x = self.tf(pil_img).unsqueeze(0).to(DEVICE)

        # Enable mixed precision for faster inference on Apple Silicon
        if DEVICE.type == "mps":
            with torch.autocast(device_type="mps", dtype=torch.float16):
                logits = self.model(x)
        else:
            logits = self.model(x)

        probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()
        pneu_prob = float(probs[self.idx_pneum])
        label_idx = self.idx_pneum if pneu_prob >= threshold else self.idx_normal

        return {
            "label": self.classes[label_idx],
            "confidence": float(probs[label_idx]),
            "probs": {c: float(p) for c, p in zip(self.classes, probs)}
        }
