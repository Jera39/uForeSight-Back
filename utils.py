def seleccionar_columnas(df):
    columnas = input("\nSelecciona las columnas a usar como características (separadas por comas): ").split(',')
    columna_objetivo = input("Selecciona la columna objetivo: ")
    columnas_categoricas = input("Selecciona las columnas categóricas (separadas por comas, si ninguna escribe 'ninguna'): ").split(',')
    if columnas_categoricas == ['ninguna']:
        columnas_categoricas = []
    return columnas, columna_objetivo, columnas_categoricas
