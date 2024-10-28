"""
Funcionalidades para criar DataLoaders do PyTorch para
dados de classificação de imagens.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageFile

NUM_WORKERS = 0  # Windows funciona melhor com 0

def check_Image(path):
  try:
    im = Image.open(path)
    return True
  except:    
    print("A imagem está truncada ou em um formato não suportado.")
    return False
ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  
  """Cria DataLoaders para treino e teste."""

  # ImageFolder para criar datasets
  train_data = datasets.ImageFolder(train_dir, transform=transform, is_valid_file=check_Image)
  test_data = datasets.ImageFolder(test_dir, transform=transform, is_valid_file=check_Image)

  # nomes das classes
  class_names = train_data.classes

  # Converte as imagens em DataLoaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names

def create_dataset_valid(path):
  valid_dataset = datasets.ImageFolder(path)

  return valid_dataset
