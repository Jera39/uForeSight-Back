#uForeSight-Back 🎯

Este repositorio contiene el back-end de uForeSight, un software de análisis y predicción de datos basado en machine learning. Esta parte se encarga de procesar los datos y realizar las predicciones usando un modelo de regresión lineal.

#🚀 Características principales

Procesamiento de datos: Limpia y prepara los datasets para su análisis.
Modelo predictivo: Utiliza un modelo de regresión lineal para realizar predicciones basadas en las características seleccionadas.
Endpoints funcionales: Configuración de rutas mediante Flask para interactuar con el front-end.
Facilidad de uso: Ideal para integrarse con cualquier aplicación de análisis de datos.

#📂 Estructura del proyecto

uForeSight-Back/
├── __pycache__/          # Caché de Python
├── ml_env/               # Entorno de machine learning (configuración)
├── data_processing.py    # Módulo para limpieza y procesamiento de datos
├── endpoints.py          # Definición de rutas para la API
├── model_training.py     # Entrenamiento y validación del modelo
├── requirements.txt      # Dependencias necesarias
├── test_flask.py         # Pruebas del servidor Flask
├── utils.py              # Funciones auxiliares

#🛠️ Tecnologías utilizadas

- Python: Lenguaje principal para el desarrollo del back-end.
- Flask: Framework ligero para la creación de APIs.
- Scikit-learn: Para implementar el modelo de regresión lineal.
- Pandas: Manejo y análisis de datos.
- NumPy: Cálculos matemáticos eficientes.

#⚙️ Instalación y configuración

##Clona este repositorio:

git clone https://github.com/TuUsuario/uForeSight-Back.git
cd uForeSight-Back

##Crea y activa un entorno virtual:

python -m venv ml_env
source ml_env/bin/activate  # En Windows: ml_env\Scripts\activate

##Instala las dependencias:

pip install -r requirements.txt

##Inicia el servidor Flask:

python endpoints.py

#🧪 Pruebas

##Para verificar que el servidor y los endpoints funcionan correctamente, ejecuta:

python test_flask.py

#📜 Licencia
