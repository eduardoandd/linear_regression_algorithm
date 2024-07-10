import pandas as pd
from numpy import *

data = {
    'Idade': [18,23,25,33,34,43,48,51,58,63,67],
    'Custo': [871,1100,1393,1654,1915,2100,2356,2698,2959,3000,3100]
}

df=pd.DataFrame(data)

class HealthInsurance:
    def __init__(self,x,y):
        self.x=x
        self.y=y

    def correlation(self):
        varx=var(self.x)
        vary=var(self.y)
        c= cov(self.x,self.y,bias=True)[0][1]

        r= c / sqrt(varx * vary)

        return r
    
    def inclination(self):
        r=self.correlation()
        stdx=std(self.x)
        stdy=std(self.y)
        m = r * (stdy / stdx)

        return m
    
    def interception(self):
        meanx=mean(self.x)
        meany=mean(self.y)
        m=self.inclination()

        b= meany - meanx * m

        return b
    
    def prevision(self, value):
        b=self.interception()
        m=self.inclination()
        v=value
        p=b+(m*v)

        return p
    


x=df['Idade'].values
y=df['Custo'].values
h=HealthInsurance(x,y)
prevision=h.prevision(54)
print(prevision)


