import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import scipy.stats as stats
import seaborn as sns

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