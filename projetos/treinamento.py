import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, FloatType
from pyspark.sql.functions import from_unixtime
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import col

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Inicializando sessão Spark
spark = SparkSession.builder.appName("deteccao-anomalia-iot-treinamento").getOrCreate()

# Schema para dados de sensores
schema = StructType([
    StructField("time", LongType(), True),
    StructField("temperatura", FloatType(), True),
    StructField("umidade", FloatType(), True),
    StructField("qualidade_do_ar", FloatType(), True),
    StructField("luz", FloatType(), True),
    StructField("som", FloatType(), True),
])

# Caminho do CSV
csv_file_path = "/opt/spark/dados/dataset_final.csv"

# --- Funções --- #

def carrega_dados(spark, file_path, schema):
    sensor_df = spark.read.csv(file_path, schema=schema, header=True)
    sensor_df = sensor_df.drop('time')
    print(f"\nTotal de registros carregados: {sensor_df.count()}\n")
    return sensor_df



def filtra_dados_normais(sensor_df_spark, k=2.0):
    
    cols = ["temperatura","umidade","qualidade_do_ar","luz","som"]
    normal_df = sensor_df_spark

    for c in cols:
        quantis = normal_df.approxQuantile(c, [0.25, 0.75], 0.01)
        q1, q3 = quantis
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        normal_df = normal_df.filter((col(c) >= lower) & (col(c) <= upper))
    
    print(f"Registros considerados normais para treino: {normal_df.count()}\n")
    return normal_df



def prepara_dados(sensor_df, scaler_model=None):
    """Cria vetor de features e aplica padronização."""
    assembler = VectorAssembler(inputCols=["temperatura","umidade","qualidade_do_ar","luz","som"],outputCol="features")
    assembled_data = assembler.transform(sensor_df)
    
    if scaler_model is None:
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
        scaler_model = scaler.fit(assembled_data)

        mean = scaler_model.mean.toArray()
        std  = scaler_model.std.toArray()

        with open("/opt/spark/modelos/scaler_params.json","w") as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
        print("Parâmetros de padronização salvos.\n")
    
    scaled_data = scaler_model.transform(assembled_data).cache()

    return scaled_data, scaler_model

def cria_modelo_anomalia(input_shape):
    """Cria LSTM autoencoder."""
    
    model = Sequential()
    
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, activation='relu', return_sequences=False))

    model.add(RepeatVector(input_shape[0]))
    
    model.add(LSTM(64, activation='relu', return_sequences=True))
    
    model.add(TimeDistributed(Dense(input_shape[1])))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# --- Fluxo principal --- #

# Carrega dados
sensor_data = carrega_dados(spark, csv_file_path, schema)


train_normal = filtra_dados_normais(sensor_data)

# Conjunto de treino: 90% dos normais
train_data = train_normal.sample(fraction=0.9, seed=42)

# Restante dos normais (10%) para teste
test_normais = train_normal.subtract(train_data)

# Anomalias: tudo que não é normal
anomalias = sensor_data.subtract(train_normal)

# Teste misto: normais + anomalias
test_misto = test_normais.union(anomalias)





# Prepara treino
train_data_prepared, scaler_model = prepara_dados(train_data)

# Prepara cada conjunto de teste usando o mesmo scaler
test_normais_prepared, _ = prepara_dados(test_normais, scaler_model)
anomalias_prepared, _ = prepara_dados(anomalias, scaler_model)
test_misto_prepared, _ = prepara_dados(test_misto, scaler_model)


def spark_para_numpy(df_prepared):
    df_pd = df_prepared.select("scaled_features").toPandas()
    return np.array(df_pd["scaled_features"].tolist()).reshape(-1, 1, 5)

X_train = spark_para_numpy(train_data_prepared)
X_test_normais = spark_para_numpy(test_normais_prepared)
X_test_anomalias = spark_para_numpy(anomalias_prepared)
X_test_misto = spark_para_numpy(test_misto_prepared)




# Cria e treina modelo
modelo_anomalia = cria_modelo_anomalia((1,5))

modelo_anomalia.fit(X_train, X_train, epochs=15, batch_size=1, verbose=2)



# Avaliação em dados normais
loss_normais = modelo_anomalia.evaluate(X_test_normais, X_test_normais, verbose=2)
print(f"Erro reconstrução - normais: {loss_normais}")

# Avaliação em anomalias
loss_anomalias = modelo_anomalia.evaluate(X_test_anomalias, X_test_anomalias, verbose=2)
print(f"Erro reconstrução - anomalias: {loss_anomalias}")

# Avaliação em teste misto
loss_misto = modelo_anomalia.evaluate(X_test_misto, X_test_misto, verbose=2)
print(f"Erro reconstrução - teste misto: {loss_misto}")




# Salva modelo
modelo_anomalia.save("/opt/spark/modelos/modelo.keras")
print("Modelo LSTM salvo com sucesso.\n")

# Encerra Spark
spark.stop()
