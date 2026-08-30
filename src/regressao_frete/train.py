%pip install scikit-learn==1.4.2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

def treinar_modelo_frete(tabela_itens):
    print("1. Carregando os dados via Spark...")
    df = spark.table(tabela_itens).toPandas()  # noqa: F821
    df.columns = df.columns.str.strip().str.lower()
    
    print(f"-> Total de registros originais no banco: {len(df)}")

    # As features que gostaríamos de usar
    features_desejadas = ['price', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
    target = 'freight_value'

    # --- O ESCUDO (Resiliência de Dados) ---
    # Verifica quais features realmente têm dados (menos de 50% de nulos)
    features_validas = []
    for col in features_desejadas:
        if col in df.columns:
            nulos = df[col].isna().sum()
            if nulos < (len(df) * 0.5): 
                features_validas.append(col)
                
    print(f"2. Features selecionadas automaticamente (com dados válidos): {features_validas}")

    # Agora limpamos os nulos apenas das features que sobraram
    df_model = df[features_validas + [target]].dropna()
    print(f"-> Base de treinamento pronta com {len(df_model)} registros.")

    X = df_model[features_validas]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("3. Treinando o Random Forest Regressor...")
    # Diminuí o max_depth para treinar mais rápido nessa prova de conceito
    modelo = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    modelo.fit(X_train, y_train)

    print("4. Avaliando o modelo...")
    predicoes = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, predicoes)
    
    print("-" * 50)
    print(f"🎉 SUCESSO! Erro Médio Absoluto (MAE): R$ {mae:.2f} de diferença no frete.")
    print("-" * 50)

    print("5. Exportando o artefato...")
    os.makedirs('artefatos', exist_ok=True)
    joblib.dump(modelo, 'artefatos/modelo_frete.joblib')
    print("Modelo salvo na pasta de artefatos e pronto para o deploy!")

if __name__ == "__main__":
    tabela_itens = 'workspace.default.olist_order_items_dataset'
    treinar_modelo_frete(tabela_itens)