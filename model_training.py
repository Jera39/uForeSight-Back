from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def dividir_datos(df, columnas, columna_objetivo):
    X = df[columnas]
    y = df[columna_objetivo]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def entrenar_modelo(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

def probar_modelo(modelo, le_encoders, columnas, valores_prueba, columna_objetivo):
    for col, val in valores_prueba.items():
        if col in le_encoders:
            val = le_encoders[col].transform([val])[0]
        else:
            val = float(val)
