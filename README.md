# Detecção de Anomalias em Tempo Real com Spark Streaming e LSTM

Este projeto implementa um sistema de detecção de anomalias em tempo real para dados de sensores IoT. A arquitetura de processamento utiliza Apache Spark Structured Streaming para lidar com fluxos de dados contínuos, enquanto uma Rede Neural LSTM identifica comportamentos atípicos.

O modelo avalia dados de sensores, calcula o erro de reconstrução e classifica cada ponto como **normal** ou **anomalia** de forma instantânea. Todo o ambiente de processamento é facilmente configurado e escalável usando Docker.

---

### Dataset
O modelo foi treinado com o dataset Anomliot IoT, disponível no Kaggle.
Link para o dataset: [Anomliot IoT](https://www.kaggle.com/datasets/hkayan/anomliot)

---

### Instalação e Execução

#### Pré-requisitos
Certifique-se de ter o Docker instalado.


### No terminal ou prompt de comando e acesse a pasta onde estão os arquivos no seu computador

### Crie e inicialize o cluster Spark
```bash
docker compose -f docker-compose.yml up -d --scale spark-worker=2
```

### No host, execute o treinamento
```bash
docker exec spark-master spark-submit --deploy-mode client ./jobs/treinamento.py
```

<img width="499" height="845" alt="Captura de tela 2025-08-26 121513" src="https://github.com/user-attachments/assets/00a9ebdc-d4a4-4f43-9b4d-5ddb8e8e297d" />
*Como mostra o resultado do treinamento, o erro de reconstrução do modelo LSTM é significativamente maior para os dados anômalos (0.159) em comparação com os dados normais (0.001), indicando que o modelo aprendeu a identificar as anomalias com sucesso.*


### No container spark-master, abra um shell para receber dados em tempo real
```bash
nc -lk 9999
```

### No host, capture dados em tempo real e gere previsões
```bash
docker exec spark-master spark-submit --deploy-mode client ./jobs/deploy.py
```

## Exemplos de Entrada
### Envie registros de sensores no formato JSON:
{"temperatura": 25.1, "umidade": 60.5, "qualidade_do_ar": 75, "luz": 628.0, "som": 145.0}
{"temperatura": 24.8, "umidade": 58.0, "qualidade_do_ar": 55, "luz": 589.0, "som": 152.0} #anômalo
{"temperatura": 27.5, "umidade": 55.2, "qualidade_do_ar": 75, "luz": 632.0, "som": 160.0}
{"temperatura": 65.5, "umidade": 12.0, "qualidade_do_ar": 75, "luz": 630.0, "som": 148.0} #anômalo
{"temperatura": 26.2, "umidade": 62.1, "qualidade_do_ar": 75, "luz": 629.0, "som": 450.0} #anômalo


<img width="1699" height="477" alt="Captura de tela 2025-08-26 135936" src="https://github.com/user-attachments/assets/2c9a6e51-d0a6-450c-ae67-aecadbc0fb1c" />
*A imagem demonstra o sistema em operação, onde os dados enviados via *nc* são processados em tempo real, e o modelo classifica cada registro como *normal* ou *anomalia*.*

