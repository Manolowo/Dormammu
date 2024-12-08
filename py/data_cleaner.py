import pandas as pd

def clean_data(df):
    # Identificar los prefijos de columnas que tienen datos mensuales
    column_prefixes = ['Col', 'ColL1TE', 'EeccInt', 'EeccNac', 'Fac', 'FacAI', 'FacAN', 'FacCCOT',
                       'FacCCPC', 'FacCI', 'FacCN', 'FacCOL', 'FacDebAtm', 'FacDebCom', 'FacPAT',
                       'FlgAct', 'FlgActAI', 'FlgActAN', 'FlgActCCOT', 'FlgActCCPC', 'FlgActCI',
                       'FlgActCN', 'FlgActCOL', 'FlgActPAT', 'PagoInt', 'PagoNac', 'Txs', 'TxsAI',
                       'TxsAN', 'TxsCCOT', 'TxsCCPC', 'TxsCI', 'TxsCN', 'TxsCOL', 'TxsDebAtm',
                       'TxsDebCom', 'TxsPAT', 'UsoL1', 'UsoL2', 'UsoLI']

    # Procesar las columnas de datos mensuales
    for prefix in column_prefixes:
        # Encontrar las columnas que comienzan con el prefijo y contienen los datos mensuales (T01 a T12)
        monthly_columns = [col for col in df.columns if str(col).startswith(prefix) and '_T' in str(col)]

        if monthly_columns:
            # Crear la nueva columna con la media
            df[prefix + '_Media'] = df[monthly_columns].mean(axis=1)

            # Eliminar las columnas originales de los meses
            df.drop(monthly_columns, axis=1, inplace=True)

    # Seleccionar solo las columnas numéricas para el análisis de outliers
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # Calcular los cuartiles y el rango intercuartílico (IQR)
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1

    # Definir los límites superior e inferior para identificar outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrar el DataFrame para eliminar outliers en las columnas numéricas
    mask = (numeric_df < lower_bound) | (numeric_df > upper_bound)

    # Eliminar solo las filas que tienen outliers en las columnas numéricas
    df_cleaned = df[~mask.any(axis=1)]

    # Rellenar valores nulos solo en columnas numéricas
    df_cleaned[numeric_df.columns] = df_cleaned[numeric_df.columns].fillna(df_cleaned[numeric_df.columns].mean())

    return df_cleaned