"""
Funções utilitárias para treinamento e salvamento de modelos em PyTorch.
"""
import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    
    # Criar diretório de destino
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Criar caminho para salvar o modelo
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name deve terminar com '.pt' ou '.pth'"
    model_save_path = target_dir_path / model_name

    # Salvar o estado do modelo
    print(f"[INFO] Salvando modelo em: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)

def load_model(model, weights_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(weights_path, map_location=device))

    return model
