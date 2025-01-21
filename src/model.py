import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# Defina as classes do seu modelo
classes = ["não TB", "TB"]

# Defina os mesmos transforms utilizados no treinamento
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class TuberculosisModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Congelar camadas conforme necessário
        # Modificar a última camada totalmente conectada
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.resnet(x)  # Logits crus (sem softmax)

# Função para carregar o modelo
def load_model(model_path="src/weights/best_model-18-01-2025.pth", device="cpu"):
    model = TuberculosisModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
