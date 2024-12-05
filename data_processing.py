import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def cargar_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def preparar_datos(df, columnas_categoricas, columna_objetivo):
    le_encoders = {col: LabelEncoder() for col in columnas_categoricas + [columna_objetivo]}
    for col in columnas_categoricas + [columna_objetivo]:
        df[col] = le_encoders[col].fit_transform(df[col])
    return df, le_encoders

def escalar_datos(df, columnas_a_escalar, columna_objetivo):
    columnas_a_escalar = [col for col in columnas_a_escalar if col != columna_objetivo]
    scaler = StandardScaler()
    df[columnas_a_escalar] = scaler.fit_transform(df[columnas_a_escalar])
    return df
