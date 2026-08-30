import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import unittest

class TestModeloFrete(unittest.TestCase):
    
    def setUp(self):
        # Cria dados simulados (Mock) idênticos ao que o modelo espera
        np.random.seed(42)
        self.dados = pd.DataFrame({
            'price': np.random.uniform(10, 1000, 50),
            'freight_value': np.random.uniform(5, 100, 50)
        })
        self.features = ['price']
        self.target = 'freight_value'

    def test_treinamento_e_previsao(self):
        """Testa se o modelo treina e cospe o formato correto de predição"""
        X = self.dados[self.features]
        y = self.dados[self.target]
        
        modelo = RandomForestRegressor(n_estimators=10, max_depth=5)
        modelo.fit(X, y)
        
        # Simulando a chegada de um novo produto que custa R$ 150.00
        novo_produto = pd.DataFrame({'price': [150.00]})
        previsao = modelo.predict(novo_produto)
        
        # Verificações (Asserts) que o GitHub vai validar:
        self.assertEqual(len(previsao), 1, "O modelo deve retornar exatamente 1 previsão")
        self.assertTrue(isinstance(previsao[0], float), "A previsão do frete deve ser um número decimal (float)")
        self.assertTrue(previsao[0] > 0, "O valor do frete não pode ser negativo")

if __name__ == '__main__':
    unittest.main()