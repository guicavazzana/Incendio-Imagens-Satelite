import os
import requests
from concurrent.futures import ThreadPoolExecutor
import random

# Defina a pasta de destino
output_dir = r'C:\Users\cavaz\OneDrive\Documentos\Faculdade\TCC\Incendio-Imagens-Satelite\dataset_satelite\train\normal'

# Certifique-se de que o diretório existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Função para baixar uma imagem aleatória
def download_image(image_number, retries=3):
    # URL do Lorem Picsum com tamanho aleatório
    width = random.randint(200, 800)  # Largura aleatória entre 200 e 800
    height = random.randint(200, 800)  # Altura aleatória entre 200 e 800
    url = f'https://picsum.photos/{width}/{height}'
    
    for attempt in range(retries):
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                file_path = os.path.join(output_dir, f'random_{image_number}.jpg')
                with open(file_path, 'wb') as out_file:
                    out_file.write(response.content)
                print(f'Imagem {image_number} baixada com sucesso.')
                break  # Sai do loop se o download for bem-sucedido
            else:
                print(f'Falha ao baixar imagem {image_number}. Status: {response.status_code}')
        except Exception as e:
            print(f'Ocorreu um erro ao baixar imagem {image_number}: {e}')

# Número total de imagens
total_images = 14

# Função principal para baixar imagens com multithreading
def download_images_concurrently():
    max_workers = 40  # Número de threads para download

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(download_image, range(1, total_images + 1))

if __name__ == '__main__':
    download_images_concurrently()
