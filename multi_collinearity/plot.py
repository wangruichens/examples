import numpy as np
import  matplotlib.pyplot as plt


x=np.array([[20.5,19.8],[21.2,20.4],[22.8,21.1],
           [18.2,23.6],[20.3,24.9],[21.8,26.7],
           [25.2,28.9],[30.7,31.3],[36.1,35.8],
           [44.3,38.2]])
a=[]
b=[]
for i in x:
    a.append(i[0])
    b.append(i[1])
c=range(20,50,3)
plt.plot(c,a,'o--',c,b,'ro--')
plt.show()