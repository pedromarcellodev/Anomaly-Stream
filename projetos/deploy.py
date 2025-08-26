import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, FloatType, LongType
from pyspark.ml.feature import VectorAssembler
from tensorflow.keras.models import load_model

# Inicializando sessão Spark
spark = SparkSession.builder.appName("deteccao-anomalia-iot-deploy").getOrCreate()

# Schema dos dados
schema = StructType([
    StructField("temperatura", FloatType(), True),
    StructField("umidade", FloatType(), True),
    StructField("qualidade_do_ar", FloatType(), True),
    StructField("luz", FloatType(), True),
    StructField("som", FloatType(), True),
])

# Carregando modelo LSTM salvo
model = load_model("/opt/spark/modelos/modelo.keras")
print("\nModelo LSTM carregado com sucesso.")


# Carregando os parâmetros de padronização
try:
    with open("/opt/spark/modelos/scaler_params.json", "r") as f:
        scaler_params = json.load(f)
        mean = np.array(scaler_params["mean"])
        std = np.array(scaler_params["std"])
    print("Parâmetros de padronização carregados com sucesso.\n")
except Exception as e:
    print(f"Erro ao carregar os parâmetros de padronização: {e}")
    exit(1)


# Função de detecção de anomalias
def detecta_anomalias(batch_df, batch_id):
    # Converter para vetor numpy
    batch_np = np.array([row['features'].toArray() for row in batch_df.collect()])

    # Verificar se o lote está vazio
    if batch_np.size == 0:
        print(f"Batch {batch_id} está vazio. Nada para processar.")
        return

     # Caso tenha uma única amostra, transformar para 2D
    if batch_np.ndim == 1:
        batch_np = batch_np.reshape(1, -1)  

    # Aplica a padronização, evitando divisão por zero
    epsilon = 1e-8
    batch_np = (batch_np - mean) / (std + epsilon)


    # Ajustar para o formato esperado pelo LSTM
    batch_np = batch_np.reshape(-1, 1, 5)  

    # Realizar a previsão com o modelo carregado
    predictions = model.predict(batch_np)

    # Calcular o erro de reconstrução
    threshold = 0.07
    anomalies = []
    for i, pred in enumerate(predictions):
        error = np.mean(np.abs(pred - batch_np[i]))
        if error > threshold:
            anomalies.append((i, error))

    
    # Exibir anomalias detectadas ou mensagem de ausência
    if anomalies:
        print(f"Anomalia detectada no batch {batch_id}: {anomalies}")
    else:
        print(f"Nenhuma anomalia detectada no batch {batch_id}.")




# Streaming via socket
streaming_data = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()


sensor_data = streaming_data \
    .selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*") \
    .withColumn("temperatura", col("temperatura").cast("float")) \
    .withColumn("umidade", col("umidade").cast("float")) \
    .withColumn("qualidade_do_ar", col("qualidade_do_ar").cast("float")) \
    .withColumn("luz", col("luz").cast("float")) \
    .withColumn("som", col("som").cast("float"))



# Preparação dos dados para o modelo
assembler = VectorAssembler(inputCols = ["temperatura", "umidade", "qualidade_do_ar", "luz", "som"], outputCol = "features")
assembled_data = assembler.transform(sensor_data)

# Streaming de detecção de anomalias
print("\nPronto para detecção de anomalias em tempo real...\n")
query = assembled_data.writeStream \
    .foreachBatch(detecta_anomalias) \
    .start()

query.awaitTermination()
spark.stop()
