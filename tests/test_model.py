import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def test_formato_dados_treinamento():
    """Garante que as features de entrada possuem o schema correto."""
    X_teste = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
    
    assert X_teste.shape[1] == 2, "O modelo deve receber exatamente 2 features"
    assert "feature1" in X_teste.columns, "A coluna 'feature1' está ausente"

def test_treinamento_e_inferencia():
    """Garante que o modelo treina e gera previsões numéricas sem erros matemáticos."""
    X_treino = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y_treino = pd.Series([10, 20, 30])
    
    modelo = RandomForestRegressor(n_estimators=10, random_state=42)
    modelo.fit(X_treino, y_treino)
    
    X_novo = pd.DataFrame({'feature1': [4], 'feature2': [7]})
    previsao = modelo.predict(X_novo)
    
    assert len(previsao) == 1, "A previsão deve retornar exatamente 1 valor"
    assert isinstance(previsao[0], float), "O output do regressor deve ser numérico (float)"