import streamlit as st
from PIL import Image
import torch
from modular.predictions import predict_single_image

st.set_page_config(page_title='Imagens Customizadas')
st.title("Imagens Customizadas para Identificar Incêndios Florestais")


# Guidance on how to use the Demo.
st.header('Como Utilizar')
st.write('Envie uma imagem de satélite de uma floresta e veja a previsão do modelo que classifica se a área possui incêndios ou não.')
# Model prediction header
st.header("Teste sua imagem abaixo")


# Get Device
device = 'GPU' if torch.cuda.is_available() else 'CPU'
st.warning(f'Previsão irá rodar via {device.upper()}')


# Get the image using file_uploader widget
image = st.file_uploader(label='Carregue uma imagem de satelite',
                 accept_multiple_files=False)

with st.spinner("Imagem sendo processada... Aguarde..."):
    if image is not None:
        # Abrir a imagem
        image = Image.open(image).convert('RGB')

        # Exibir a imagem
        st.image(image)

        # Enviar a imagem para previsão
        predicted_class = predict_single_image(image)

        if predicted_class == "Imagem Inválida":
            st.error(f'Classe identificada: {predicted_class}. Utilize uma imagem válida de satélite.')
        else:
            st.info(f'Classe identificada: {predicted_class}')


