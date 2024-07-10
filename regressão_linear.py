from numpy import *

class LinearRegression:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.__correlation_coefficient=self.__correlacao() #1
        self.__inclination=self.__inclinacao() #2
        self.__intercept = self.__interceptacao()

    
    def __correlacao(self):
        covariacao=cov(self.x,self.y,bias=True)[0][1]
        var_x=var(self.x)
        var_y=var(self.y)

        return covariacao / sqrt(var_x * var_y)

    def __inclinacao(self):
        stdx=std(self.x) #desv padrão
        stdy=std(self.y)
        return self.__correlation_coefficient * (stdy/ stdx)

    def __interceptacao(self):
        mediax=mean(self.x)
        mediay=mean(self.y)
        b=mediay-mediax * self.__inclination
        return b

    def previsao(self,valor):
        return self.__intercept + (self.__inclination * valor)
    

x=array([1,2,3,4,5])
y=array([2,4,6,8,10])
lr=LinearRegression(x,y)
previsao=lr.previsao(6)
print(previsao)

