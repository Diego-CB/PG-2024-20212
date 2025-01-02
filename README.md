# Proyecto: Desarrollo y Evaluación de Modelos de Análisis de Sentimientos en Imágenes y Textos con IA Explicable y Propuesta de Integración para Estudios de Mercado

## 1. Descripción
El proyecto desarrolla un sistema integral de análisis de emociones que combina técnicas de visión por computadora y procesamiento de lenguaje natural. Utiliza un detector de rostros en tiempo real basado en clasificadores en cascada con filtros de Haar, y un modelo de redes neuronales convolucionales (CNN) para clasificar expresiones faciales. Además, se implementa un modelo de análisis de sentimientos en texto que complementa el análisis emocional. Para asegurar la transparencia, se aplican técnicas de Explainable AI, permitiendo comprender las decisiones de los modelos.

## 2. Instrucciones de Instalación

### Requisitos previos
- python 3.10
- docker instalado
- contar con alguna videocamara en la computadora para poder ejecutar el pipeline de detección de expresiones faciales en tiempo real.

### Instalación

Como primer punto se deben construir las siguientes imágenes de docker:

- docker build -t torch:cuda .
- docker build -t model -f ./Dockerfile_serveModel .
- docker build -t text -f ./Dockerfile_TextAPI .

Luego deberá crear un ambiente virtual de python e instalar dependencias:

 1. `python -m venv venv`
 2. `./venv/scrips/activate`
 3. `pip install -r requirements.txt`

### Ejecución


**Pipeline de detección de expresiones faciales**

1. Levantar servidor local corriendo el siguiente script de powershell [runModel](src/runModel.ps1) o con `docker run -it -v ${PWD}:/repo/ --gpus all --rm -p 5000:5000 model`
2. Iniciar ambiente virtual con `./venv/scrips/activate`
3. Iniciar el stram de video con `python ./streamWebcam`

**Análisis de sentimiento en texto**

1. Levantar servidor local corriendo el siguiente script de powershell [runText](src/runText.ps1) o con `docker run -it --rm -v ${PWD}:/repo/ -p 8000:8000 --name textApi -t text`

## 3. Demo
Puedes acceder a la demostración aquí: [Demo](demo/demo.mp4)

## 4. Informe Final

Puedes acceder al informe aquí: [Informe Final](docs/Trabajo_de_graduación___Diego_Córdova.pdf)
