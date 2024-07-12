#tem uma performace muito boa em alguns casos
# utiliza probabilidade para criar modelo de machine learning
#olha os dados históricos e calcula a chance do que a gente ta querendo classificar ser enfluenciada 

# Naive Bayes
# Análisa cada atributo da classe de forma independete

# passo 1 - calcular a probabilidade condicional da classe
# passo 2 - calcular a probabilidade condicional dos atributos com a classe
# passo 3 - calculo de probabilidade posterior

import pandas as pd
from sklearn.model_selection import train_test_split # dividir os dados em train e test
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder # transforma dados categoricos em informações numericas
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report #análise de métricas
from yellowbrick.classifier import ConfusionMatrix # gera matriz de confusão

#Label encoder: transformar var em elementos númericos
#dividir os dados em traine e teste
#criar modelo
# fazer previsões
# avaliar performace
#metrica de confusão

# tratando os dados
df = pd.read_csv('ecommerce_sales_analysis.csv')
df=df.drop(columns=['product_name'])
df.shape

y=df.iloc[:,5].values
X=df.iloc[:,1:4].values

#Etapa label encoder
label_encoder=LabelEncoder()
type(label_encoder)

for i in range(X.shape[1]):
    if X[:,i].dtype == 'object':
        X[:,i] = label_encoder.fit_transform(X[:,i])

    
#Etapa dividir os dados em treino e teste
# 70 % para treinamento e 30% para teste
x_treinamento,x_teste,y_treinamento,y_teste=train_test_split(X,y,test_size=0.3,random_state=1)


#Etapa criação de modelo
modelo = GaussianNB()
modelo.fit(x_treinamento,y_treinamento) # treinando o modelo

#PREVISÃO
previsao=modelo.predict(x_teste)