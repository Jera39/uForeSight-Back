# from data_processing import cargar_dataset, preparar_datos, escalar_datos
# from model_training import dividir_datos, entrenar_modelo, probar_modelo
# from utils import seleccionar_columnas

# # Flujo principal
# if __name__ == "__main__":
#     # Cargar el dataset
#     file_path = input("Ingresa la ruta del dataset (archivo CSV): ")
#     df = cargar_dataset(file_path)

#     # Seleccionar columnas
#     columnas, columna_objetivo, columnas_categoricas = seleccionar_columnas(df)

#     # Preparar datos
#     df, le_encoders = preparar_datos(df, columnas_categoricas, columna_objetivo)

#     # Escalar datos
#     columnas_a_escalar = [col for col in columnas if col not in columnas_categoricas + [columna_objetivo]]
#     df = escalar_datos(df, columnas_a_escalar, columna_objetivo)

#     # Dividir datos
#     X_train, X_test, y_train, y_test = dividir_datos(df, columnas, columna_objetivo)

#     # Entrenar modelo
#     modelo, accuracy = entrenar_modelo(X_train, y_train, X_test, y_test)
#     print(f"Precisión del modelo: {accuracy:.2f}")

#     # Probar modelo
#     valores_prueba = {}
#     print("\nIngresa los valores de prueba:")
#     for col in columnas:
#         val = input(f"Valor para {col}: ")
#         valores_prueba[col] = val
#     predicciones = probar_modelo(modelo, le_encoders, columnas, valores_prueba, columna_objetivo)
#     print(f"\nPredicciones: {predicciones}")

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from io import BytesIO



app = Flask(__name__)
CORS(app)  # Permite peticiones desde cualquier origen (útil para el frontend)

state = {
    'df': None,
    'columnas': None,
    'columna_objetivo': None,
    'columnas_categoricas': None,
    'modelo': None,
    'le_encoders': None
}

# Variable global para almacenar el archivo temporal
uploaded_file_content = None  # Aquí guardaremos el contenido del archivo
original_dataset = None  # Aquí guardaremos el archivo original

@app.route('/upload', methods=['POST'])
def upload_dataset():
    global uploaded_file_content, original_dataset  # Usamos variables globales
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se encontró el archivo en la solicitud'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'El archivo no tiene nombre'}), 400

        # Guardar el contenido del archivo en memoria
        uploaded_file_content = file.read()
        state['dataset_name'] = file.filename

        try:
            file.seek(0)  # Asegurarnos de leer desde el inicio
            state['df'] = pd.read_csv(BytesIO(uploaded_file_content))
            # Guardar una copia del dataset original para el home
            original_dataset = state['df'].copy()
        except Exception as e:
            return jsonify({'error': f'Error al leer el archivo: {str(e)}'}), 400

        return jsonify({
            'message': f"Archivo '{file.filename}' subido exitosamente.",
            'columns': state['df'].columns.tolist(),
            'dataset_name': state['dataset_name'],
            'preview': state['df'].head(10).to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({'error': f'Error inesperado: {str(e)}'}), 500




@app.route('/select-columns', methods=['POST'])
def select_columns():
    try:
        # Recibir las columnas seleccionadas desde el cliente
        data = request.json
        state['columnas'] = data['features']
        state['columna_objetivo'] = data['target']
        state['columnas_categoricas'] = data.get('categorical', [])

        # Regenerar los LabelEncoders con los datos actuales
        state['le_encoders'] = {
            col: LabelEncoder().fit(state['df'][col]) for col in state['columnas_categoricas'] + [state['columna_objetivo']]
        }

        # Reiniciar el modelo para evitar inconsistencias
        state['modelo'] = None

        return jsonify({
            'message': 'Columnas seleccionadas correctamente',
            'features': state['columnas'],
            'target': state['columna_objetivo'],
            'categorical': state['columnas_categoricas']
        })
    except Exception as e:
        return jsonify({'error': f'Error al seleccionar columnas: {str(e)}'}), 400



@app.route('/train-model', methods=['POST'])
def train_model():
    try:
        # Validar que el DataFrame esté cargado
        if state['df'] is None:
            return jsonify({'error': 'El dataset no está cargado. Por favor, carga un dataset antes de entrenar el modelo.'}), 400

        # Validar que las columnas estén definidas
        if state['columnas'] is None or state['columna_objetivo'] is None:
            return jsonify({'error': 'Las columnas no están seleccionadas. Por favor, selecciona columnas antes de entrenar el modelo.'}), 400

        # Validar que las columnas existan en el DataFrame
        missing_columns = [col for col in state['columnas'] + [state['columna_objetivo']] if col not in state['df'].columns]
        if missing_columns:
            return jsonify({'error': f'Las siguientes columnas no existen en el dataset: {missing_columns}'}), 400

        print("Paso 1: Columnas y datos validados correctamente.")

        from data_processing import preparar_datos, escalar_datos
        from model_training import dividir_datos, entrenar_modelo

        # Preparar los datos
        df, le_encoders = preparar_datos(state['df'], state['columnas_categoricas'], state['columna_objetivo'])
        print("Paso 2: Datos preparados correctamente.")
        print(df.head())

        # Escalar los datos numéricos
        columnas_a_escalar = [col for col in state['columnas'] if col not in state['columnas_categoricas'] + [state['columna_objetivo']]]
        print(f"Columnas a escalar: {columnas_a_escalar}")

        # Si no hay columnas para escalar, saltar el escalado
        if columnas_a_escalar:
            df = escalar_datos(df, columnas_a_escalar, state['columna_objetivo'])
            print("Paso 3: Datos escalados correctamente.")
        else:
            print("Paso 3: No hay columnas numéricas para escalar. Saltando escalado.")

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = dividir_datos(df, state['columnas'], state['columna_objetivo'])
        print("Paso 4: Datos divididos correctamente.")
        print(f"X_train:\n{X_train.head()}")
        print(f"y_train:\n{y_train.head()}")

        # Verificar si los conjuntos están vacíos
        if X_train.empty or y_train.empty:
            return jsonify({'error': 'Los datos de entrenamiento están vacíos. Por favor, revisa las columnas seleccionadas y el dataset.'}), 400

        # Entrenar el modelo
        modelo, accuracy = entrenar_modelo(X_train, y_train, X_test, y_test)
        print("Paso 5: Modelo entrenado correctamente.")
        print(f"Accuracy: {accuracy}")

        # Guardar el modelo y encoders en el estado
        state['modelo'] = modelo
        state['le_encoders'] = le_encoders
        state['accuracy'] = accuracy

        return jsonify({
            'message': 'Modelo entrenado correctamente',
            'accuracy': accuracy,
            'target': state.get('columna_objetivo')  # Envía la columna objetivo si está disponible
        })

    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        return jsonify({'error': f'Error al entrenar el modelo: {str(e)}'}), 400



@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validar que el modelo esté entrenado
        if state['modelo'] is None or state['columnas'] is None:
            return jsonify({'error': 'El modelo no está entrenado. Por favor, entrena el modelo antes de realizar una predicción.'}), 400

        data = request.json
        print(f"Datos recibidos para la predicción: {data}")

        # Validar que todas las columnas necesarias estén presentes
        missing_columns = [col for col in state['columnas'] if col not in data]
        if missing_columns:
            return jsonify({'error': f'Faltan las siguientes columnas en los datos enviados: {missing_columns}'}), 400

        # Preparar los datos para la predicción
        input_data = {}
        for col in state['columnas']:
            val = data[col]
            print(f"Procesando columna: {col}, valor: {val}")

            if col in state['le_encoders']:
                try:
                    val = state['le_encoders'][col].transform([val])[0]
                    print(f"Valor transformado para {col}: {val}")
                except ValueError:
                    valid_values = list(state['le_encoders'][col].classes_)
                    return jsonify({
                        'error': f"El valor '{val}' no es válido para la columna '{col}'. Valores válidos: {valid_values}"
                    }), 400

            input_data[col] = val

        print(f"Datos preparados para la predicción: {input_data}")
        df_input = pd.DataFrame([input_data])

        # Realizar la predicción
        prediction = state['modelo'].predict(df_input)
        print(f"Predicción bruta: {prediction}")

        # Invertir la codificación de la columna objetivo si es categórica
        if state['columna_objetivo'] in state['le_encoders']:
            prediction = state['le_encoders'][state['columna_objetivo']].inverse_transform(prediction)
            print(f"Predicción transformada: {prediction}")

        return jsonify({
            'message': 'Predicción realizada con éxito',
            'prediction': prediction[0]
        })
    except Exception as e:
        print(f"Error durante la predicción: {str(e)}")
        return jsonify({'error': f'Error al realizar la predicción: {str(e)}'}), 400


@app.route('/reset-columns', methods=['POST'])
def reset_columns():
    global uploaded_file_content  # Usamos el archivo temporal guardado
    try:
        if uploaded_file_content is None:
            return jsonify({'error': 'No hay un archivo cargado previamente. Por favor, sube un archivo primero.'}), 400

        # Leer nuevamente el archivo desde la memoria
        try:
            state['df'] = pd.read_csv(BytesIO(uploaded_file_content))
        except Exception as e:
            return jsonify({'error': f'Error al recargar el archivo: {str(e)}'}), 400

        # Reiniciar todos los estados excepto el dataset
        state['columnas'] = None
        state['columna_objetivo'] = None
        state['columnas_categoricas'] = None
        state['modelo'] = None
        state['le_encoders'] = None

        return jsonify({'message': 'El estado se ha reiniciado correctamente y el archivo se ha recargado.'})

    except Exception as e:
        return jsonify({'error': f'Error al reiniciar el estado: {str(e)}'}), 500

@app.route('/dataset', methods=['GET'])
def get_dataset():
    try:
        if state['df'] is None:
            return jsonify({'error': 'No hay un dataset cargado. Por favor, sube un archivo primero.'}), 400
        
        # Convertir el DataFrame a un formato JSON para enviar al frontend
        columns = state['df'].columns.tolist()
        data = state['df'].head(100).to_dict(orient='records')  # Solo enviamos una vista previa de las primeras 100 filas
        
        return jsonify({
            'columns': columns,
            'data': data,
            'dataset_name': state.get('dataset_name', 'No hay un dataset cargado')
        })
    except Exception as e:
        return jsonify({'error': f'Error al obtener el dataset: {str(e)}'}), 500


@app.route('/home-dataset', methods=['GET'])
def get_home_dataset():
    try:
        if original_dataset is None:
            return jsonify({'error': 'No hay un dataset cargado. Por favor, sube un archivo primero.'}), 400
        
        # Convertir el DataFrame original a un formato JSON para enviar al frontend
        columns = original_dataset.columns.tolist()
        data = original_dataset.head(100).to_dict(orient='records')  # Solo enviamos una vista previa de las primeras 100 filas

        return jsonify({
            'columns': columns,
            'data': data,
            'dataset_name': state.get('dataset_name', 'No hay un dataset cargado')
        })
    except Exception as e:
        return jsonify({'error': f'Error al obtener el dataset para el home: {str(e)}'}), 500



@app.route('/results', methods=['GET'])
def get_model_status():
    try:
        if state['modelo'] is None:
            return jsonify({'error': 'No hay un modelo entrenado. Por favor, entrena el modelo primero.'}), 400

        return jsonify({
            'message': 'Modelo entrenado y listo.',
            # 'accuracy': state['modelo'].accuracy if hasattr(state['modelo'], 'accuracy') else None
            'accuracy': state.get('accuracy'),
            'columns': state['columnas']
        })
    except Exception as e:
        return jsonify({'error': f'Error al obtener el estado del modelo: {str(e)}'}), 500

@app.route('/get-configuration', methods=['GET'])
def get_configuration():
    try:
        return jsonify({
            'features': state.get('columnas', []),
            'target': state.get('columna_objetivo', None),
            'categorical': state.get('columnas_categoricas', [])
        })
    except Exception as e:
        return jsonify({'error': f'Error al obtener la configuración: {str(e)}'}), 500

@app.route('/select-columns', methods=['GET'])
def get_selected_columns():
    try:
        return jsonify({
            'features': state.get('columnas', []),
            'target': state.get('columna_objetivo', None),
            'categorical': state.get('columnas_categoricas', [])
        })
    except Exception as e:
        return jsonify({'error': f'Error al obtener la configuración: {str(e)}'}), 500

@app.route('/unique-values', methods=['GET'])
def get_unique_values():
    try:
        column = request.args.get('column')
        if not column or column not in original_dataset:  # Usamos el dataset original
            return jsonify({'error': 'Columna no válida'}), 400

        # Obtener los valores únicos de la columna del dataset original
        unique_values = original_dataset[column].unique().tolist()
        return jsonify({'unique_values': unique_values})
    except Exception as e:
        return jsonify({'error': f'Error al obtener valores únicos: {str(e)}'}), 500



@app.route('/metrics', methods=['GET', 'POST'])
def manage_metrics():
    if request.method == 'POST':
        data = request.json
        state.setdefault('metrics', []).append(data['metric'])
        return jsonify({'message': 'Métrica agregada', 'metrics': state['metrics']})
    elif request.method == 'GET':
        return jsonify({'metrics': state.get('metrics', [])})


if __name__ == '__main__':
    app.run(debug=True)