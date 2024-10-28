import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title='Introdução')

st.header('Classificação e Análise de Incêndios Florestais Usando Imagens de Satélite')

st.markdown('<div style="text-align: justify;">Os incêndios florestais são um desastre natural devastador que representa ameaças significativas tanto à vida humana quanto ao meio ambiente. Esses eventos catastróficos, frequentemente impulsionados por fatores como mudanças climáticas e atividade humana, têm aumentado em frequência e intensidade nos últimos anos. Portanto, é crucial desenvolver abordagens inovadoras para monitorar, prever e mitigar incêndios florestais. Uma dessas abordagens é o uso de imagens de satélite para classificação de incêndios florestais, um projeto que possui imensa importância e oferece inúmeros benefícios.</div>', unsafe_allow_html=True)
st.write('')
st.markdown('<div style="text-align: justify;">Em conclusão, o projeto de classificação de incêndios florestais usando imagens de satélite é de suma importância devido ao seu potencial para melhorar a detecção precoce, a conscientização situacional, a previsão e o monitoramento ambiental de incêndios florestais. Os benefícios que oferece, incluindo segurança pública aprimorada, redução de custos, escalabilidade e cobertura global, tornam-no uma ferramenta valiosa na luta contra incêndios florestais. Além disso, este projeto se alinha aos esforços mais amplos para enfrentar os desafios crescentes impostos pelas mudanças climáticas e seus impactos na atividade de incêndios florestais. Ao aproveitar o poder da tecnologia de satélite, podemos dar passos significativos em direção à mitigação dos efeitos devastadores dos incêndios florestais e à proteção tanto das comunidades humanas quanto do meio ambiente.</div>', unsafe_allow_html=True)

st.write('')

img = Image.open(r'./images//forest_fires.jpg')
st.image(img)
