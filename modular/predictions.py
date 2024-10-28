"""
Funções utilitárias para fazer previsões.
"""
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn

from typing import List, Tuple
from modular.utils import load_model

from PIL import Image
from pathlib import Path

# Definir dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# INICIALIZA STREAMLIT

# Nomes das classes para repositório
class_names_satellite = ['Imagem Inválida', 'satelite']  # Para validação de imagens de satélite
class_names = ['Normal (sem incêndio)', 'Incêndio']  # Para previsão de incêndio

# Inicializar modelos
model_loaded_satellite = torchvision.models.efficientnet_b0().to(device)
model_loaded_satellite.eval()

model_loaded = torchvision.models.efficientnet_b0().to(device)
model_loaded.eval()

# Modificar o classificador final para ambos os modelos
model_loaded.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=2)  # Nomes das classes codificados
).to(device)

model_loaded_satellite.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=2)  # Nomes das classes codificados
).to(device)

# Carregar os pesos dos modelos pré-treinados
model_loaded_satellite = load_model(model_loaded_satellite, Path(r"./model/EfficientNet_b0-Classificador_Satelite.pt"))
model_loaded = load_model(model_loaded, Path(r"./model/EfficientNet_b0-Classificador_Fogo.pt"))

# Pesos de transformações padrão
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT

# Obter transformações a partir dos pesos (essas são as transformações usadas para treinar um determinado conjunto de pesos)
automatic_transforms = weights.transforms()

# Função para prever e plotar imagem
def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    device: torch.device = device,
):
    # Abrir imagem
    img = Image.open(image_path)

    # Criar transformação para a imagem (se não existir uma)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # Prever na imagem
    model.to(device)
    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)
        target_image_pred = model(transformed_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Plotar imagem com rótulo previsto e probabilidade
    plt.figure()
    plt.imshow(img)
    plt.title(
        f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}"
    )
    plt.axis(False)

# Função para prever uma única imagem
def predict_single_image(image, y_label=None):  # O y_label é opcional

    # Transformar a imagem
    transformed_image = automatic_transforms(image)

    # Adicionar dimensão extra à imagem
    pred_image = torch.unsqueeze(transformed_image, dim=0)
    pred_image = pred_image.to(device)

    # Validação de imagem de satélite
    with torch.inference_mode():
        y_logits_satellite = model_loaded_satellite(pred_image)
        y_pred_prob_satellite = torch.argmax(y_logits_satellite, dim=1).item()
        predicted_class_satellite = class_names_satellite[y_pred_prob_satellite]

    # Se a imagem for inválida, retornar diretamente
    if predicted_class_satellite == 'Imagem Inválida':
        return predicted_class_satellite
    else:
    # Caso a imagem seja válida, realizar a previsão de incêndio
        with torch.inference_mode():
            y_logits = model_loaded(pred_image)
            y_pred_prob = torch.argmax(y_logits, dim=1).item()
            predicted_class = class_names[y_pred_prob]

    # Retornar a classe prevista
    if y_label is None:
        return predicted_class
    else:
        return [predicted_class, predicted_class == y_label]
