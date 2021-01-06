# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 20:44:11 2020

@author: abdo
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Perceptron function
def perceptron(x, y, z, eta, t):
    '''
    Input Parameters:
        x: entrées
        y: outputs
        z: fonction d'activation threshold
        eta: taux d'apprentissage
        t: nombre des iterations
    '''
    
    # initialisation des coefficients
    w = np.zeros(len(x[0]))      
    n = 0                        
    
    while n < t: 
        for i in range(0, len(x)): 
             
            # la multiplication des x par w
            f = np.dot(x[i], w)
           
            # fonction d'activation 
            if f >= z:                               
                p = 1.                            
            else:                                 
                p = 0.
            
            # mise a jour des coefficients
            for j in range(0, len(w)):             
                w[j] = w[j] + eta*(y[i]-p)*x[i][j]
                
        n += 1
        
    return w


#     x0  x1  x2
x = [[1., 0., 0.],
     [1., 0., 1.],
     [1., 1., 0.],
     [1., 1., 1.]]



y =[1.,
    1.,
    1.,
    0.]

z = 0.0
eta = 0.1
t = 50

print("les coefficients sont:")
print(perceptron(x, y, z, eta, t))


#################################################################

df = pd.read_csv('C:/Users/abdo/Desktop/dataset.csv')
plt.scatter(df.values[:,1], df.values[:,2], c = df['3'], alpha=0.8)

df = df.values  
                
np.random.seed(5)
np.random.shuffle(df)


train = df[0:int(0.7*len(df))]
test = df[int(0.7*len(df)):int(len(df))]

x_train = train[:, 0:3]
y_train = train[:, 3]

x_test = test[:, 0:3]
y_test = test[:, 3]


print(perceptron(x_train, y_train, z, eta, t))


from sklearn.linear_model import Perceptron

# training the sklearn Perceptron
model = Perceptron(random_state=None, eta0=0.1, shuffle=False, fit_intercept=False)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)


model.coef_





































