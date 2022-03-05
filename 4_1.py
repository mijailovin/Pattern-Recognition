import numpy as np
import scipy
from scipy import io
import matplotlib.pyplot as plt

from numpy.linalg import *

import random 
%matplotlib qt
# %matplotlib inline
'''
1. deo: po N=500 odbiraka, L=4 (broj klasa)
'''

#%% Generisanje podataka
np.random.seed(50)
N=500
L=4

#centi tih klasa (mean)
# C=np.array([(-7,3), (7,-2), (7,5), (0,5)])
C=np.array([(-9,4), (7,-2), (7,9), (0,5)])

cov=np.eye(2)
X=[]
for i in range(L):
    Xi=np.random.multivariate_normal(C[i], cov, N).T
    X.append(Xi)

plt.figure()
for i in range(L):
    plt.plot(X[i][0,:], X[i][1,:],'x')
plt.legend(['$K_1$', '$K_2$', '$K_3$', '$K_4$'])
plt.xlabel('$x_1$')
plt.xlabel('$x_2$')
#%% C-mean

plt.close('all')
np.random.seed()

Y_all=np.empty((2,L*N))
for i in range(L):
    Y_all[:,i*N:(i+1)*N]=X[i]

perm=np.random.permutation(L*N)
Y_all=Y_all[:,perm]

Y=[]
for i in range(L):
    Y.append(Y_all[:, i*N:(i+1)*N])

plt.figure()
for i in range(L):
    plt.plot(Y[i][0,:], Y[i][1,:],'x')
plt.axis('equal')
plt.legend(['$K_1$', '$K_2$', '$K_3$', '$K_4$'])
plt.xlabel('$x_1$')
plt.xlabel('$x_2$')

#Y je lista koja sadrzi cetri matrice sa odbircima u 
#pretpostavljenim klasama
#ovo provo je inicijalna klasterizacija

# C-mean

M=np.zeros((2,L))
for i in range(L):
    M[:,i]=np.mean(Y[i], axis=1)

print(M)
reklas=1
lmax=20
l=0
n=N*np.ones((4,), dtype=int) #br. elem u nekoj klasi

while (l<lmax and reklas==1):
    reklas=0
    #za novu reklasterizaciju
    Y_pom=[]
    for i in range(L): #L*N za svaku klasu
        Y_pom.append( np.zeros( (2,L*N), dtype=float ) )
    n_pom=np.zeros((L,), dtype=int)
    # prolazi se kroz svaki odbirak    
    for i in range(L):
        for j in range(n[i]):
            Yj=Y[i][:, j:j+1]
            #sad treba da se uporedi sa svakim od centara
            d=np.zeros((L,))
            for k in range(L):
                d[k]=np.sum((Yj-M[:,k:k+1])**2)
            best=np.argmin(d)
            if (best!=i):
                reklas=1
            Y_pom[best][:,n_pom[best]]=Y[i][:,j]
            n_pom[best]+=1
            
            # print(i, best, d)
    # print('iteracija:', l, n_pom)
    
    n=n_pom
    for i in range(L):
        Y[i]=Y_pom[i][:,0:n_pom[i]]
        M[:,i]=np.mean(Y[i], axis=1)
    l+=1
    plt.figure();
    plt.show()
    for i in range(L):
        plt.plot(Y[i][0,:], Y[i][1,:],'x')
    plt.axis('equal')
    plt.legend(['$K_1$', '$K_2$', '$K_3$', '$K_4$'])
    plt.xlabel('$x_1$')
    plt.xlabel('$x_2$')
    # plt.pause(0.5);
    
print(l)
    

#%% Srednji broj potrebnih iteracija
iter=np.zeros((10, ))
for s in range(10):

    Y_all=np.empty((2,L*N))
    for i in range(L):
        Y_all[:,i*N:(i+1)*N]=X[i]

    perm=np.random.permutation(L*N)
    Y_all=Y_all[:,perm]

    Y=[]
    for i in range(L):
        Y.append(Y_all[:, i*N:(i+1)*N])

    for i in range(L):
        plt.plot(Y[i][0,:], Y[i][1,:],'x')

    #Y je lista koja sadrzi cetri matrice sa odbircima u 
    #pretpostavljenim klasama
    #ovo provo je inicijalna klasterizacija

    # C-mean

    M=np.zeros((2,L))
    for i in range(L):
        M[:,i]=np.mean(Y[i], axis=1)

    reklas=1
    lmax=20
    l=0
    n=N*np.ones((4,), dtype=int) #br. elem u nekoj klasi

    while (l<lmax and reklas==1):
        reklas=0
        #za novu reklasterizaciju
        Y_pom=[]
        for i in range(L): #L*N za svaku klasu
            Y_pom.append( np.zeros( (2,L*N), dtype=float ) )
        n_pom=np.zeros((L,), dtype=int)
        # prolazi se kroz svaki odbirak    
        for i in range(L):
            for j in range(n[i]):
                Yj=Y[i][:, j:j+1]
                #sad treba da se uporedi sa svakim od centara
                d=np.zeros((L,))
                for k in range(L):
                    d[k]=np.sum((Yj-M[:,k:k+1])**2)
                best=np.argmin(d)
                if (best!=i):
                    reklas=1
                Y_pom[best][:,n_pom[best]]=Y[i][:,j]
                n_pom[best]+=1
        
        n=n_pom
        for i in range(L):
            Y[i]=Y_pom[i][:,0:n_pom[i]]
            M[:,i]=np.mean(Y[i], axis=1)
        l+=1
    iter[s]=l

#%% 
plt.figure()
plt.stem(np.arange(1,11), iter)
plt.title('Broj potrebnih iteracija')
plt.xlabel('Različite početne klasterizacije')
