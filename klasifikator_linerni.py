import numpy as np
from numpy.linalg import *

#mi imamo 3 klase, ovo se poziva za svaki par klasa 1-2, 1-3, 2-3
def num_metod_2(K1, K2):
    N1=K1.shape[1]
    N2=K2.shape[1]
    
    M1=np.reshape(np.mean(K1, axis=1), (2,1))
    M2=np.reshape(np.mean(K2, axis=1), (2,1))
    S1=np.cov(K1)
    S2=np.cov(K2)
    
    s=np.arange(0, 1, 0.05)
    Neps_s=np.zeros((len(s),))
    v0_opt_s=np.zeros((len(s),))
    for i in range(len(s)):
        V=inv(s[i]*S1+(1-s[i])*S2)@(M2-M1)
        
        Y1=V.T@K1
        Y2=V.T@K2 # ovo je 1D
        Y1=Y1.flatten()
        Y2=Y2.flatten()
        Y=np.concatenate((Y1, Y2)) ###
        Y=np.sort(Y)
        v0=np.zeros((N1+N2-1))
        Neps=np.zeros((N1+N2-1,))
        for j in range(N1+N2-1):
            v0[j]=-(Y[j]+Y[j+1])/2
            for k in range(0,N2):
                if (Y2[k]<-v0[j]):
                    Neps[j]+=1
            for k in range(0,N1):
                if (Y1[k]>-v0[j]):
                    Neps[j]+=1
            
        Neps_s[i]=np.min(Neps)
        v0_opt_s[i]=v0[np.argmin(Neps)]
        
        print(s[i])
    
    Neps_opt=np.min(Neps_s)
    v0_opt=v0_opt_s[np.argmin(Neps_opt)]
    s_opt=s[np.argmin(Neps_opt)]

    
    return s_opt, v0_opt, Neps_opt, M1, M2, S1, S2

def num_metod_3(K1, K2):
    N1=K1.shape[1]
    N2=K2.shape[1]
    N1t=round(0.8*N1)
    N2t=round(0.8*N2)
    
    M1=np.reshape(np.mean(K1[:, :N1t], axis=1), (2,1))
    M2=np.reshape(np.mean(K2[:, :N2t], axis=1), (2,1))
    S1=np.cov(K1[:, :N1t])
    S2=np.cov(K2[:, :N2t])
    
    s=np.arange(0, 1, 0.05)
    Neps_s=np.zeros((len(s),))
    v0_opt_s=np.zeros((len(s),))
    for i in range(len(s)):
        V=inv(s[i]*S1+(1-s[i])*S2)@(M2-M1)
        
        Y1=V.T@K1[:, N1t: ]
        Y2=V.T@K2[:, N2t: ]  # ovo je 1D
        Y1=Y1.flatten()
        Y2=Y2.flatten()
        Y=np.concatenate((Y1, Y2)) ###
        Y=np.sort(Y)
        
        v0=np.zeros((N1+N2-N1t-N2t-1, ))
        Neps=np.zeros((N1+N2-N1t-N2t-1, ))
        for j in range(N1+N2-N1t-N2t-1):
            # print(j)
            v0[j]=-(Y[j]+Y[j+1])/2
            for k in range(0,N2-N2t):
                if (Y2[k]<-v0[j]):
                    Neps[j]+=1
            for k in range(0,N1-N1t):
                if (Y1[k]>-v0[j]):
                    Neps[j]+=1

        Neps_s[i]=np.min(Neps)
        v0_opt_s[i]=v0[np.argmin(Neps)]
        
        print(s[i])
    
    Neps_opt=np.min(Neps_s)
    v0_opt=v0_opt_s[np.argmin(Neps_opt)]
    s_opt=s[np.argmin(Neps_opt)]

    
    return s_opt, v0_opt, Neps_opt, M1, M2, S1, S2
