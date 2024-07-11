import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import scipy.stats as stats
import seaborn as sns

#Análise de correlação
#Análise de reseduais
# previsão
#gerar gráficos

base = pd.read_csv('mt_cars.csv')
base=base.drop(['Unnamed: 0'],axis=1)

corr=base.corr() # gera um objeto com uma matriz das correlações
sns.heatmap(corr,cmap='coolwarm',annot=True,fmt='.2f')

column_pairs=[('mpg','cyl'),('mpg','disp'),('mpg','hp'),('mpg','wt'),('mpg','drat'),('mpg','vs')]
n_plots=len(column_pairs)

fig,axes=plt.subplots(nrows=n_plots,ncols=1,figsize=(6,4 * n_plots))

for i, pair in enumerate(column_pairs):
    x_col,y_col=pair
    sns.scatterplot(x=x_col,y=y_col,data=base,ax=axes[i])
    axes[i].set_title(f'{x_col} vs {y_col}')

plt.tight_layout()
plt.show()


# análisar as variáveis independetes para prever o consumo
# analisar os indicadores (reseduais)
# avaliar a performace do modelo


# ============ CRIAÇÃO DO MODELO ============
#aic 156.6 bic 162.5 (performace)
modelo = sm.ols('mpg ~ wt + disp + hp', data=base) #variável dependete na esquerda e independente na esquerda
modelo=modelo.fit() #cria o modelo
modelo.summary()

# ============ ANÁLISE DE RESEDUAIS ===========
residuos=modelo.resid
plt.hist(residuos, bins=20)
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.title('Histograma de Resíduos')
plt.show()

stats.probplot(residuos, dist='norm',plot=plt)
plt.title('Q-Q plot de residuos')
plt.show()

# ============ TESTE DE SHAPIRO ===========
#shapiro-wilk(teste estatístico) quanto mais próximo de 1 melhor, porém não é relevante.
#p-value(teste de hipótese)
# h0 - dados normalmente distribuidos
# p <=0.05 não há(provavel) h0
# P > 0.05 é(provavel) h0

stat,pval = stats.shapiro(residuos)
print(f'Shapiro-wilk statistica: {stat:.3f}, p-value:{pval:.3f}')



# ============ NOVO MODELO2 ===========
# aic 165.1 bic 169.5
modelo2= sm.ols(formula='mpg ~ disp + cyl', data=base)
modelo2=modelo2.fit()
modelo2.summary()

# ============ ANÁLISE RESEDUAIS2 ===========
residuos2=modelo2.resid
plt.hist(residuos2,bins=20)
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.title('Análise resíduos disp + cyl')
plt.show()

stats.probplot(residuos2,dist='norm',plot=plt)
plt.title('Q-Q plot de resíduos')
plt.show()

# ============ TESTE DE SHAPIRO2 ===========
stat,pval=stats.shapiro(residuos2)
print(f'Shapiro-wilk statistica: {stat:.3f}, p-value:{pval:.3f}')



# ============ MY VERSION ===========

df = pd.read_csv('data.csv')
df=df.drop('Country',axis=1)

corr=df.corr()
sns.heatmap(corr,cmap='coolwarm',annot=True,fmt='.2f')

column_pairs=[('EconomicQuality','Governance'),('EconomicQuality','InvestmentEnvironment'),('EconomicQuality','EnterpriseConditions'),('EconomicQuality','MarketAccessInfrastructure'),('EconomicQuality','Education'),('EconomicQuality','NaturalEnvironment')]
n_plots=len(column_pairs)
fig,axes= plt.subplots(nrows=n_plots,ncols=1, figsize=(6,4 * n_plots))

for i,pair in enumerate(column_pairs):
    x_col,y_col=pair
    sns.scatterplot(x=x_col,y=y_col,data=df,ax=axes[i])
    axes[i].set_title(f'{x_col} vs {y_col}')

plt.tight_layout()
plt.show()

#MODELO - ANÁLISE DE PERFORMACE E RESIDUAIS

#aic 1059 bic 1071
modelo3= sm.ols(formula='EconomicQuality ~ InvestmentEnvironment + EnterpriseConditions + Education',data=df)
# modelo3= sm.ols(formula='EconomicQuality ~ Governance + NaturalEnvironment',data=df)
modelo3=modelo3.fit()
modelo3.summary()

residuos3=modelo3.resid
plt.hist(residuos3,bins=20)
plt.xlabel('Residuos')
plt.ylabel('Frequência')
plt.title('Histograma de Residuos')

stats.probplot(residuos3, dist='norm', plot=plt)
plt.title('Q-Q Plot de Residuos')
plt.show()

#TESTE DE SHAPIRO 
stat,pval=stats.shapiro(residuos3)
print(f'shapiro-wilk: {stat} , p-value: {pval}')