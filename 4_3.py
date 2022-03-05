import numpy as np
import scipy
from scipy import io
import matplotlib.pyplot as plt

from numpy.linalg import *

import random 
%matplotlib qt
# %matplotlib inline

#%% Generisanje podataka
np.random.seed(50)
N=500
L=4

#centi tih klasa (mean)
# C=np.array([(-9,3), (7,-2), (15, 9), (0,10)])
C=np.array([(-7,2), (7,-2), (0,5), (7,5)])

cov=np.eye(2)
X=[]
for i in range(L):
    Xi=np.random.multivariate_normal(C[i], cov, N).T
    X.append(Xi)

plt.figure()
for i in range(L):
    plt.plot(X[i][0,:], X[i][1,:],'x')

#%% Generisanje podataka 2
N=500
L=2
# np.random.seed(5)
M11=np.array([[0],[0]])
S11=np.array([[2, 0.5],\
              [0.5, 4.5]])
M12=np.array([[19],[-1]])
S12=np.array([[5.5, -0.7],\
              [-0.7, 2.8]])
M21=np.array([[8],[7]])
S21=np.array([[4, -1],\
              [-1, 11]])
M22=np.array([[13],[12]])
S22=np.array([[5, -2.7],\
              [-2.7, 9.8]])
X1=np.zeros((2,N))
for i in range(N):
    r=np.random.rand(1)
    if (r<0.6):
        X1[:,i]=np.random.multivariate_normal(M11.flatten(), S11)
    else:
        X1[:,i]=np.random.multivariate_normal(M12.flatten(), S12)
        
X2=np.zeros((2,N)) 
for i in range(N):
    r=np.random.rand(1)
    if (r<0.55):
        X2[:,i]=np.random.multivariate_normal(M21.flatten(), S21).T
    else:
        X2[:,i]=np.random.multivariate_normal(M22.flatten(), S22).T

plt.figure()
plt.plot(X1[0,:],X1[1,:],'x')
plt.plot(X2[0,:],X2[1,:],'x')
plt.axis('equal') 

X=[]
X.append(X1)
X.append(X2)
#%% Generisanje podataka 3
N=500
L=2
# np.random.seed(5)
M11=np.array([[1],[1]])
S11=np.array([[4, 1.1],\
              [1.1, 2]])
M12=np.array([[6],[4]])
S12=np.array([[3, -0.8],\
              [-0.8, 1.5]])
M21=np.array([[7],[-4]])
S21=np.array([[2, 1.1],\
              [1.1, 4]])
M22=np.array([[6],[-1]])
S22=np.array([[3, 0.8],\
              [0.8, 0.5]])
X1=np.zeros((2,N))
for i in range(N):
    r=np.random.rand()
    if (r<0.6):
        X1[:,i]=np.random.multivariate_normal(M11.flatten(), S11).T
    else:
        X1[:,i]=np.random.multivariate_normal(M12.flatten(), S12).T
        
X2=np.zeros((2,N)) 
for i in range(N):
    r=np.random.rand()
    if (r<0.55):
        X2[:,i]=np.random.multivariate_normal(M21.flatten(), S21).T
    else:
        X2[:,i]=np.random.multivariate_normal(M22.flatten(), S22).T

plt.figure()
plt.plot(X1[0,:],X1[1,:],'x')
plt.plot(X2[0,:],X2[1,:],'x')
plt.legend(['$K_1$','$K_2$'])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')

X=[]
X.append(X1)
X.append(X2)
#%% Kvadratna dekompozicija
# np.random.seed(251)
plt.close('all')

koliko=50 # toliko odbiraka svake klasa je odmah na pocetku dobro smesteno
Y_all=np.empty((2,L*(N-koliko)))
klase=np.zeros((1,))

for i in range(L):
    Y_all[:,i*(N-koliko):(i+1)*(N-koliko)]=X[i][:,koliko:]
    # klase=np.concatenate((klase, i*np.ones(N,)))
klase=klase[1:] # sluzi posle da se vidi kojoj klasi pripada svaki odbirak

perm=np.random.permutation(L*(N-koliko))
Y_all=Y_all[:,perm]
# klase=klase[perm]
klase=np.zeros((L*N,))


Y=[]
for i in range(L):
    Y.append(np.concatenate((Y_all[:, i*(N-koliko):(i+1)*(N-koliko)], X[i][:,:koliko]), axis= 1  ))

Y_all=np.zeros((2, L*N))
for i in range(L):
    Y_all[:,i*N:(i+1)*N]=Y[i]

plt.figure()
for i in range(L):
    plt.plot(Y[i][0,:], Y[i][1,:],'x')
plt.legend(['$K_1$','$K_2$'])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')


#pocetna procena Pi, Mi, Sigma_i
P=np.zeros((L, ))
M=[]
S=[]
Mi=np.zeros((2,1))
for i in range(L):
    P[i]=Y[i].shape[1]/(N*L)
    Mi=np.reshape(np.mean(Y[i], axis=1), (2,1))
    M.append(Mi)
    S.append(np.cov(Y[i]))

l=0; lmax=100; reklas=1;
while (l<lmax and reklas==1):
    reklas=0
    klases=klase.copy()
    iz=[]
    for i in range(L):
        iz.append(np.zeros((N*L,)))
        for j in range(N*L):
            Xj=Y_all[:,j:j+1]
            iz[i][j]=0.5*(Xj-M[i]).T@inv(S[i])@(Xj-M[i])\
                +0.5*np.log(det(S[i]))-0.5*np.log(P[i])
    
    for j in  range(N*L):
        mink=1e6
        for i in range(L):
            if (iz[i][j]<mink):
                mink=iz[i][j]
                klase[j]=i
        if (klase[j]!=klases[j]):
            reklas=1
            
    # novi parametri
    Y=[]
    for i in range(L):
        Y.append(Y_all[:,klase==i])
    for i in range(L):
        P[i]=Y[i].shape[1]/(N*L)
        M[i]=np.reshape(np.mean(Y[i], axis=1), (2,1))
        S[i]=np.cov(Y[i])
        
    plt.figure()
    for i in range(L):
        plt.plot(Y[i][0,:], Y[i][1,:],'x')
    plt.legend(['$K_1$','$K_2$'])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.axis('equal')
    plt.show()
    l+=1 
    print(l, P)

#%% Srednji broj iteracija

iter=np.zeros((20, ))
for s in range(20):
    koliko=50 # toliko odbiraka svake klasa je odmah na pocetku dobro smesteno
    Y_all=np.empty((2,L*(N-koliko)))
    klase=np.zeros((1,))
    
    for i in range(L):
        Y_all[:,i*(N-koliko):(i+1)*(N-koliko)]=X[i][:,koliko:]
    
    perm=np.random.permutation(L*(N-koliko))
    Y_all=Y_all[:,perm]
    klase=np.zeros((L*N,))
    
    
    Y=[]
    for i in range(L):
        Y.append(np.concatenate((Y_all[:, i*(N-koliko):(i+1)*(N-koliko)], X[i][:,:koliko]), axis= 1  ))
    
    Y_all=np.zeros((2, L*N))
    for i in range(L):
        Y_all[:,i*N:(i+1)*N]=Y[i]
    
    #pocetna procena Pi, Mi, Sigma_i
    P=np.zeros((L, ))
    M=[]
    S=[]
    Mi=np.zeros((2,1))
    for i in range(L):
        P[i]=Y[i].shape[1]/(N*L)
        Mi=np.reshape(np.mean(Y[i], axis=1), (2,1))
        M.append(Mi)
        S.append(np.cov(Y[i]))
    
    l=0; lmax=100; reklas=1;
    while (l<lmax and reklas==1):
        reklas=0
        klases=klase.copy()
        iz=[]
        for i in range(L):
            iz.append(np.zeros((N*L,)))
            for j in range(N*L):
                Xj=Y_all[:,j:j+1]
                iz[i][j]=0.5*(Xj-M[i]).T@inv(S[i])@(Xj-M[i])\
                    +0.5*np.log(det(S[i]))-0.5*np.log(P[i])
        
        for j in  range(N*L):
            mink=1e6
            for i in range(L):
                if (iz[i][j]<mink):
                    mink=iz[i][j]
                    klase[j]=i
            if (klase[j]!=klases[j]):
                reklas=1
                
        # novi parametri
        Y=[]
        for i in range(L):
            Y.append(Y_all[:,klase==i])
        for i in range(L):
            P[i]=Y[i].shape[1]/(N*L)
            M[i]=np.reshape(np.mean(Y[i], axis=1), (2,1))
            S[i]=np.cov(Y[i])
            
        l+=1 
    print('l', end='')
    iter[s]=l
#%% Crtanje broja potrebnih iteracija

plt.figure()
plt.stem(np.arange(1,21), iter)
plt.title('Broj potrebnih iteracija')
plt.xlabel('Različite početne klasterizacije')
