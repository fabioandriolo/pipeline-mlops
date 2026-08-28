import os
import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import tarfile

print("1. Treinando o modelo para gerar o artefato atualizado...")
X = pd.DataFrame({'feature1': [1, 2, 3, 4], 'feature2': [5, 6, 7, 8]})
y = pd.Series([10, 20, 30, 40])
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X, y)

joblib.dump(modelo, 'model.joblib')
with tarfile.open('model.tar.gz', 'w:gz') as tar:
    tar.add('model.joblib')

print("2. Enviando para o AWS S3 (Autenticado via GitHub Secrets)...")
bucket_name = 'meu-projeto-mlops-data'
s3 = boto3.client('s3')
s3.upload_file('model.tar.gz', bucket_name, 'artefatos/model.tar.gz')

print("3. Preparando o ambiente de inferência...")
os.makedirs('code', exist_ok=True)
with open('code/requirements.txt', 'w') as f:
    f.write("scikit-learn==1.4.2\njoblib\n")

codigo_inferencia = """import joblib
import os

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, 'model.joblib'))
"""
with open('code/inference.py', 'w') as f:
    f.write(codigo_inferencia)

print("4. Conectando à AWS e gerenciando a infraestrutura...")
boto_session = boto3.Session(region_name="us-west-2")
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role_arn = "arn:aws:iam::808632059053:role/service-role/AmazonSageMaker-ExecutionRole-20260821T162079"
model_uri = f's3://{bucket_name}/artefatos/model.tar.gz'
endpoint_name = "api-preditiva-automatizada"

# Limpeza preventiva para permitir atualizações contínuas
sm_client = boto_session.client('sagemaker')
try:
    print("Buscando endpoint antigo para substituição...")
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
    print("Endpoint antigo deletado.")
except Exception:
    print("Nenhum endpoint antigo encontrado. Seguindo para a criação...")

print("5. Iniciando o Deploy no SageMaker (ml.t2.medium)...")
sklearn_model = SKLearnModel(
    model_data=model_uri,
    role=role_arn,
    entry_point='inference.py',
    source_dir='code',
    framework_version='1.4-2',
    py_version='py3',
    sagemaker_session=sagemaker_session
)

sklearn_model.deploy(
    instance_type='ml.t2.medium',
    initial_instance_count=1,
    endpoint_name=endpoint_name
)

print(f"Deploy finalizado! A API {endpoint_name} está no ar!")