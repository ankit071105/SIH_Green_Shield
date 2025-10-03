from PIL import Image
import torch
from torchvision import transforms
import os
from typing import Optional, Tuple

# CLASSES will be updated if you train on PlantVillage; default mapping below
CLASSES = ['healthy','leaf_blight','leaf_spot','powdery_mildew']

_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def _build_mobilenet(num_classes):
    import torchvision.models as models
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    return model

def _build_fallback(num_classes):
    import torch.nn as nn
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.classifier = nn.Linear(32, num_classes)
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)
    return SimpleCNN()

def load_model(weights_path: Optional[str] = None):
    num_classes = len(CLASSES)
    # Prefer torchvision MobileNet if available; else fallback
    try:
        model = _build_mobilenet(num_classes)
    except Exception:
        model = _build_fallback(num_classes)
    if weights_path and weights_path != '' and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location='cpu')
        try:
            model.load_state_dict(state)
        except Exception:
            # saved object might be full model; try returning it
            try:
                return state
            except Exception:
                pass
    model.eval()
    return model

def predict_torch(model, image: Image.Image) -> Tuple[str, float]:
    x = _transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
        return CLASSES[int(idx)], float(conf)
