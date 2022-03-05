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
# C=np.array([(-9,4), (7,-2), (7,9), (0,5)])
C=np.array([(-7,2), (7,-2), (7,5), (0,5)])

cov=np.eye(2)
X=[]
for i in range(L):
    Xi=np.random.multivariate_normal(C[i], cov, N).T
    X.append(Xi)

plt.figure()
for i in range(L):
    plt.plot(X[i][0,:], X[i][1,:],'x')

#%% Maximum likelihood clustering
# np.random.seed(50)


Y_all=np.empty((2,L*N))
for i in range(L):
    Y_all[:,i*N:(i+1)*N]=X[i]

perm=np.random.permutation(L*N)
Y_all=Y_all[:,perm]

Lp=5 #pretpostavljeni broj odbiraka
Y=[]
Np=L*N//Lp # broj odbiraka po klasama u pocetnoj random klasterizaciji
for i in range(Lp):
    Y.append(Y_all[:, i*Np:(i+1)*Np])

plt.figure()
for i in range(Lp):
    plt.plot(Y[i][0,:], Y[i][1,:],'x')

#pocetna procena Pi, Mi, Sigma_i
P=np.zeros((Lp, ))
M=[]
S=[]
Mi=np.zeros((2,1))
for i in range(Lp):
    P[i]=Y[i].shape[1]/(N*L)
    Mi=np.reshape(np.mean(Y[i], axis=1), (2,1))
    M.append(Mi)
    S.append(np.cov(Y[i]))


f=[] #1. clan f1(Xj), 2. clan f2(Xj) ...
for i in range(Lp):
    f.append(np.zeros((N*L, )))
    for j in range(L*N):
        Xi=Y_all[:,j:j+1]
        f[i][j]=1/(2*np.pi*det(S[i])**0.5)*np.exp(-0.5*(Xi-M[i]).T@inv(S[i])@(Xi-M[i]))
q=[]   
fall=np.zeros_like(f[0])    
for i in range(Lp):
    fall+=P[i]*f[i]
    
for i in range(Lp):
    q.append(P[i]*f[i]/fall)


reklas=1
l=0
lmax=200
maxq=0; delta=0.01
while (l<lmax and reklas==1):
    print(l)
    Ss=S.copy()
    Ms=M.copy()
    Ps=P.copy()
    qs=q.copy()
       
    for i in range(Lp):
        P[i]=1/(L*N)*np.sum(qs[i])
            
    for i in range(Lp):
        for j in range(N*L):
            Xi=Y_all[:,j:j+1]
            M[i]+=qs[i][j]*Xi

        M[i]/=N*L*Ps[i]
        M[i]=np.reshape(M[i], (2,1))
 
    for i in range(Lp):
        for j in range(N*L):
            Xi=Y_all[:,j:j+1]
            S[i]+=qs[i][j]*(Xi-Ms[i])@(Xi-Ms[i]).T
        S[i]/= N*L*Ps[i]
        
    for i in range(Lp):
        for j in range(L*N):
            Xi=Y_all[:,j:j+1]
            f[i][j]=1/(2*np.pi*det(S[i])**0.5)*np.exp(-0.5*(Xi-M[i]).T@inv(S[i])@(Xi-M[i]))
            
    fall=np.zeros_like(f[0])
    for i in range(Lp):
        fall+=P[i]*f[i]
    q=[]
    for i in range(Lp):
        q.append(P[i]*f[i]/fall)
    l+=1
    
    for i in range(Lp):
        if (P[i]<0.0001):
            l=lmax
    
    # usov za zaustavlajnje
    maxq=0
    for i in range(Lp):
        maxi=np.max(q[i]-qs[i])
        if (maxi>maxq):
            maxq=maxi
    if (maxq<delta):
        reklas=0
        
    print(maxq)

#%% Klasterizacija 
plt.close('all')
q_ind=np.zeros((L*N, ), dtype=int)
for j in range(L*N):
    max_=1e-10
    ind=-1
    for i in range(Lp):
        if (q[i][j]>max_):
            ind=i
            max_=q[i][j]
    q_ind[j]=ind
Y_=[]
for i in range(Lp):
    Y_.append(Y_all[:, q_ind==i])
    
plt.figure()
for i in range(Lp):
    plt.plot(Y_[i][0,:],Y_[i][1,:],'x')
plt.axis('equal')
plt.legend(['$K_1$', '$K_2$', '$K_3$', '$K_4$', '$K_5$'])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')  
    