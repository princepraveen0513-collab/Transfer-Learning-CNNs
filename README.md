# Transfer Learning with PyTorch — 3‑Part Project (Step‑by‑Step)

This guide combines **TransferLearning_Part1/2/3** into a single, reproducible workflow. You’ll start with a **feature extractor**, then **fine‑tune** selected layers, and finally evaluate/export the best model.

---

## 📁 Repository Layout
```
.
├── TransferLearning_Part1.ipynb    # Step 1: Feature Extractor (frozen backbone + new head)
├── TransferLearning_Part2.ipynb    # Step 2: Fine‑Tuning (unfreeze selected layers)
├── TransferLearning_Part3.ipynb    # Step 3: Evaluation, Export, and Inference
└── README.md                        # This combined guide
```

---

## 🧰 Environment & Setup
**Dependencies:** `torch`, `torchvision`, `pillow`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`

Install:
```bash
pip install -U torch torchvision torchaudio pillow numpy pandas scikit-learn matplotlib
```

**Backbones detected:** resnet18  
**Pretrained weights used:** Yes  
**Backbone frozen (feature extractor mode):** Yes (in Step 1)  
**Classifier head replaced:** Yes  
**Transforms observed:** Normalize, Resize, ToTensor  
**Device:** CUDA (if available) (auto‑detected in notebooks)

---

## ✅ Step 1 — Feature Extractor (Part 1)
**Notebook:** `TransferLearning_Part1.ipynb`

- Load dataset with ImageNet‑style transforms (e.g., `Resize`/`CenterCrop` → `ToTensor` → `Normalize`).  
- Initialize a pretrained backbone (e.g., resnet18).  
- **Freeze** backbone params and **replace the classifier head** to match `num_classes`.  
- Train only the new head; record baseline metrics.

Run:
```bash
jupyter notebook "TransferLearning_Part1.ipynb"
```

---

## ⚙️ Step 2 — Fine‑Tuning Strategy (Part 2)
**Notebook:** `TransferLearning_Part2.ipynb`

- **Unfreeze** deeper layers (or the entire backbone) and set a **lower LR** for backbone vs head.  
- Optionally add a **scheduler** (e.g., StepLR, CosineAnnealingLR, OneCycleLR) and **weight decay**.  
- Train for additional epochs; monitor validation metrics to avoid overfitting.  
- Save the best checkpoint.

Run:
```bash
jupyter notebook "TransferLearning_Part2.ipynb"
```

---

## 📊 Step 3 — Evaluation & Export (Part 3)
**Notebook:** `TransferLearning_Part3.ipynb`

- Evaluate on the test set (Accuracy / Precision / Recall / F1 if computed).  
- Generate qualitative results (sample predictions).  
- Export the trained model (`torch.save`) and class mapping for inference.

Run:
```bash
jupyter notebook "TransferLearning_Part3.ipynb"
```

---

## 📈 Results (best observed in outputs)

- **Best Accuracy:** **91.550**

---

## 🚀 Inference (after training)
```python
import torch
from torchvision import transforms
from PIL import Image

# load
model = torch.load("checkpoints/best_model.pth", map_location="cpu")
model.eval()

# preprocessing (match training!)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

img = Image.open("path/to/image.jpg").convert("RGB")
x = preprocess(img).unsqueeze(0)  # [1, C, H, W]
with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)
pred = probs.argmax(dim=1).item()
print("Predicted class id:", pred)
```

---

## 🧭 Next Steps
- Try a different backbone (e.g., ResNet50 / EfficientNet / ConvNeXt).  
- Use **discriminative learning rates** (lower for backbone, higher for head).  
- Add **data augmentation** (RandAugment / AutoAugment) for robustness.  
- Consider **mixed precision** (`torch.cuda.amp`) for speed on GPU.  
- Export to **TorchScript** or **ONNX** for deployment.

