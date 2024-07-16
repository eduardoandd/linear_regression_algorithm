#tem uma performace muito boa em alguns casos
# utiliza probabilidade para criar modelo de machine learning
#olha os dados históricos e calcula a chance do que a gente ta querendo classificar ser enfluenciada 

# Naive Bayes
# Análisa cada atributo da classe de forma independete

# passo 1 - calcular a probabilidade condicional da classe
# passo 2 - calcular a probabilidade condicional dos atributos com a classe

import pandas as pd
from sklearn.model_selection import train_test_split # dividir os dados em train e test
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder # transforma dados categoricos em informações numericas
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report #análise de métricas
from yellowbrick.classifier import ConfusionMatrix # gera matriz de confusão

#tratamento
df = pd.read_csv('index.csv')
df=df.drop(columns=['datetime','card'])


y=df.iloc[:,-1].values
X=df.iloc[:,:-2].values

#Etapa label encoder
label_encoder=LabelEncoder()
for i in range(X.shape[1]):
    if X[:,i].dtype=='object':
        X[:,i]=label_encoder.fit_transform(X[:,i])

#Etapa dividir os dados em treino e teste
# 70 % para treinamento e 30% para teste
x_training,x_test,y_training,y_test=train_test_split(X,y,test_size=0.3,random_state=1)

#Criação de modelo
modelo=GaussianNB()
modelo.fit(X,y)

#previsao
previsao=modelo.predict()

#medição de performace
accuracy = accuracy_score(y_test,previsao)
precision=precision_score(y_test,previsao,average='weighted')
recall= recall_score(y_test,previsao,average='weighted')
f1= f1_score(y_test,previsao,average='weighted')

report= classification_report(y_test,previsao)