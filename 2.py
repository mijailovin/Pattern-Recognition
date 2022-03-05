import numpy as np
import scipy
from scipy import io
import matplotlib.pyplot as plt

from numpy.linalg import *

import random 
%matplotlib qt
# %matplotlib inline


plt.close('all')
#%% Generisanje podataka
N=500
np.random.seed(15)

M11=np.array([[1],[1]])
S11=np.array([[0.3, 0],\
              [0, 0.1]])
M12=np.array([[2],[2]])
S12=np.array([[0.7, -0.5],\
              [-0.5, 0.7]])
M21=np.array([[2],[6]])
S21=np.array([[2, -0.5],\
              [-0.5, 1]])
M22=np.array([[5],[2]])
S22=np.array([[0.9, -0.7],\
              [-0.7, 2]])
X1=np.zeros((2,N))
for i in range(N):
    # X1[:,i]=np.random.multivariate_normal(M11.flatten(), S11)

    r=np.random.rand(1)
    if (r<0.5):
        X1[:,i]=np.random.multivariate_normal(M11.flatten(), S11)
    else:
        X1[:,i]=np.random.multivariate_normal(M12.flatten(), S12)
        
X2=np.zeros((2,N)) 
for i in range(N):
    r=np.random.rand(1)
    if (r<0.5):
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

#%% d^2 krive

x=np.linspace(-4, 10, 200)
y=np.linspace(-2, 11, 200)

f1=np.zeros((x.shape[0], y.shape[0]))
f2=np.zeros((x.shape[0], y.shape[0]))
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        X=np.array( [ [x[i]], [y[j]] ] )
        f11=1/(2*np.pi*np.linalg.det(S11)**0.5)*np.exp(-0.5*(X-M11).T@np.linalg.inv(S11)@(X-M11))
        f12=1/(2*np.pi*np.linalg.det(S12)**0.5)*np.exp(-0.5*(X-M12).T@np.linalg.inv(S12)@(X-M12))
        f1[i,j]=0.5*f11+0.5*f12;
        
        f21=1/(2*np.pi*np.linalg.det(S21)**0.5)*np.exp(-0.5*(X-M21).T@np.linalg.inv(S21)@(X-M21))
        f22=1/(2*np.pi*np.linalg.det(S22)**0.5)*np.exp(-0.5*(X-M22).T@np.linalg.inv(S22)@(X-M22))
        f2[i,j]=0.5*f21+0.5*f22;

#### kao transponiovanje ????

d2=np.array([2, 5, 8]) # te nivoe hocemo
prag1=np.max(f1)*np.exp(-0.5*d2);
prag2=np.max(f2)*np.exp(-0.5*d2);

plt.figure()
plt.contour(x,y,f1.T, levels=prag1[-1::-1], linewidths=3, cmap='jet')
plt.contour(x,y,f2, levels=prag2[-1::-1], linewidths=3, cmap='jet')
plt.plot(X1[0,:],X1[1,:],'x')
plt.plot(X2[0,:],X2[1,:],'x')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(['$K_1$','$K_2$'])
plt.axis('equal')

#%% Bayes-ov klasifikator

#msm da bi trebalo na nekim test podacima

h=np.zeros((x.shape[0], y.shape[0]))
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        h[i,j]=-np.log(f1[i,j]/f2[i,j])

plt.figure()
plt.contour(x,y,h.T,levels=[0], linewidths=3)
plt.plot(X1[0,:],X1[1,:],'x')
plt.plot(X2[0,:],X2[1,:],'x')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(['$K_1$', '$K_2$'])
plt.axis('equal')

#%% Procena greske
import seaborn as sn

def greska(x, y, X1, X2, h, N, prag_novi=0):
    conf=np.zeros((2,2))
    for k in range(N):
        i=np.argmin(np.abs(x-X1[0,k]))
        j=np.argmin(np.abs(y-X1[1,k]))
        if (h[i,j]>prag_novi): #pogresno
            conf[1,0]+=1
        else:
            conf[0,0]+=1
    for k in range(N):
        i=np.argmin(np.abs(x-X2[0,k]))
        j=np.argmin(np.abs(y-X2[1,k]))
        if (h[i,j]>prag_novi): #tacno
            conf[1,1]+=1
        else:
            conf[0,1]+=1
    # print(conf)
    plt.figure()
    sn.heatmap(conf, annot=True, xticklabels=['$K_1$', '$K_2$'], yticklabels=['$K_1$', '$K_2$'], fmt='g', cmap='Blues')
    
    e1=conf[1,0]/N
    e2=conf[0,1]/N
    e=0.5*e1+0.5*e2
    print(e1, e2, e)
    return e
#####
e=greska(x, y, X1, X2, h, N)

#%% sad na nekim nezavisnim test podacima
# np.random.seed(11)
plt.close('all')
Nt=round(N*0.3)
print(Nt)
X1_test=np.zeros((2,Nt))
for i in range(Nt):
    X1_test[:,i]=np.random.multivariate_normal(M11.flatten(), S11)

    r=np.random.rand(1)
    if (r<0.5):
        X1_test[:,i]=np.random.multivariate_normal(M11.flatten(), S11)
    else:
        X1_test[:,i]=np.random.multivariate_normal(M12.flatten(), S12)
        
X2_test=np.zeros((2,Nt)) 
for i in range(Nt):
    r=np.random.rand(1)
    if (r<0.5):
        X2_test[:,i]=np.random.multivariate_normal(M21.flatten(), S21).T
    else:
        X2_test[:,i]=np.random.multivariate_normal(M22.flatten(), S22).T

greska(x, y, X1_test, X2_test, h, Nt)

plt.figure()
plt.contour(x,y,h.T,levels=[0], linewidths=3)
plt.plot(X1_test[0,:],X1_test[1,:],'x')
plt.plot(X2_test[0,:],X2_test[1,:],'x')
plt.legend(['$K_1$','$K_2$'])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')
#%% Neyman-Pearson-ov test
        
plt.close('all')
#treba izabrati epsilon2
epsilon2=0.02

# nije nula prag, nego h(x)>-ln(mi)
#treba mi histogram h|w2

h1=np.zeros((N,))
for k in range(N):
    i=np.argmin(np.abs(x-X1[0,k]))
    j=np.argmin(np.abs(y-X1[1,k]))
    h1[k]=h[i,j]
h2=np.zeros((N,))
for k in range(N):
    i=np.argmin(np.abs(x-X2[0,k]))
    j=np.argmin(np.abs(y-X2[1,k]))
    h2[k]=h[i,j]

#sad histograme za ova dva h1, h2
h2_hist, bin_edges= np.histogram(h2, bins=50, density=True)

fig, ax=plt.subplots(1,2, figsize=(12,5))
ax[0].hist(h1, bins=50, density=True); ax[0].set_title('$f_h(h|\omega_1)$');
ax[0].set_xlabel('$h$')
ax[1].hist(h2, bins=50, density=True); ax[1].set_title('$f_h(h|\omega_2)$');
ax[1].set_xlabel('$h$')
# treba da se integrali
integral=0
i=0
dh=bin_edges[1]-bin_edges[0]
while (integral<epsilon2):
    integral+=h2_hist[i]*dh
    print(integral)
    i+=1
print(i)
prag_novi=(bin_edges[i]+bin_edges[i-1])/2
# prag_novi=(bin_edges[i-1]
print('-ln(mi)=', prag_novi)



# sada sa ovim novim pragom
plt.figure()
plt.plot(X1[0,:],X1[1,:],'x')
plt.plot(X2[0,:],X2[1,:],'x')
plt.contour(x,y,h.T,levels=[ prag_novi], linewidths=3, cmap='jet')
plt.legend(['$K_1$','$K_2$'])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')

#sada da se proceni greska za nov klasifikator
greska(x, y, X1, X2, h, N, prag_novi)

#%% sad na nekim nezavisnim test podacima- Neyman
# np.random.seed(11)
plt.close('all')
Nt=10000
print(Nt)
X1_test=np.zeros((2,Nt))
for i in range(Nt):
    r=np.random.rand()
    if (r<0.5):
        X1_test[:,i]=np.random.multivariate_normal(M11.flatten(), S11).T
    else:
        X1_test[:,i]=np.random.multivariate_normal(M12.flatten(), S12).T
        
X2_test=np.zeros((2,Nt)) 
for i in range(Nt):
    r=np.random.rand()
    if (r<0.5):
        X2_test[:,i]=np.random.multivariate_normal(M21.flatten(), S21).T
    else:
        X2_test[:,i]=np.random.multivariate_normal(M22.flatten(), S22).T

greska(x, y, X1_test, X2_test, h, Nt, prag_novi)

plt.figure()
plt.plot(X1_test[0,:],X1_test[1,:],'x')
plt.plot(X2_test[0,:],X2_test[1,:],'x')
plt.contour(x,y,h.T,levels=[0, prag_novi], linewidths=3)
plt.legend(['$K_1$','$K_2$'])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')

#%% Wald-ov test
plt.close('all')
#kolike su greske
e1=1e-30
e2=1e-30
a=-np.log((1-e1)/e2)
b=-np.log(e1/(1-e2))
print('a=',a,'; b=',b)
# za pokazivanje
# a=-100
# b=100

plt.figure()
for l in range(10):
    perm=np.random.permutation(N)
    Y1=X1
    Y1=Y1[:,perm]
    #odbirci prve klase dolaze
    Sm=np.zeros((N,))
    k=0
    while (Sm[k]>a):
        i=np.argmin(np.abs(x-Y1[0,k]))
        j=np.argmin(np.abs(y-Y1[1,k]))
        Sm[k+1]=Sm[k]+h[i,j]
        k+=1
    if (l<9):
        plt.plot(range(0, k+1), Sm[0:k+1], 'blue', label='_nolegend_')
    else:
        plt.plot(range(0, k+1), Sm[0:k+1], 'blue')

for l in range(10):
    perm=np.random.permutation(N)

    Y2=X2
    Y2=Y2[:,perm]
    #odbirci druge klase dolaze
    Sm=np.zeros((N,))
    k=0
    while (Sm[k]<b):
        i=np.argmin(np.abs(x-Y2[0,k]))
        j=np.argmin(np.abs(y-Y2[1,k]))
        Sm[k+1]=Sm[k]+h[i,j]
        k+=1
    if (l<9):
         plt.plot(range(0, k+1), Sm[0:k+1], 'red', label='_nolegend_')
    else:
        plt.plot(range(0, k+1), Sm[0:k+1], 'red')
    
plt.axhline(y=a, color='black', linestyle='--')
plt.axhline(y=b, color='black', linestyle='--')
plt.legend(['$\omega_1$', '$\omega_2$'])
# plt.legend(handles=[, line2])
    
#%% Zavisnost broja potrebnih odbiraka od e1/e2 (za 1. klasu)
plt.close('all')
E1=np.logspace(-50, -0.01,  200)
E2=np.logspace(-50, -0.01,  200)

e2=1e-20
k_mean=np.zeros((E1.shape[0], ))
m1=np.zeros((E1.shape[0], )); eta1=np.mean(h1)
for i in range(E1.shape[0]):
    e1=E1[i]; 
    a=-np.log((1-e1)/e2)
    b=-np.log(e1/(1-e2))
    # print(a, b)
    for l in range(20):
        perm=np.random.permutation(N)
        Y1=X1
        Y1=Y1[:,perm]
        #odbirci prve klase dolaze
        Sm=np.zeros((N,))
        k=0
        while (Sm[k]>a):
            i1=np.argmin(np.abs(x-Y1[0,k]))
            j1=np.argmin(np.abs(y-Y1[1,k]))
            Sm[k+1]=Sm[k]+h[i1,j1]
            k+=1
        k_mean[i]+=k-1
    k_mean[i]/=20  
    m1[i]=(a*(1-e1)+b*e1)/eta1


plt.figure()
plt.semilogx(E1, k_mean)
plt.semilogx(E1, m1)
plt.xlabel('$\epsilon_1$')
plt.legend(['$\hat{m}_1(\epsilon_1)$','$m_1(\epsilon_1)$'])

e1=1e-20
k_mean=np.zeros((E2.shape[0], ))
m1=np.zeros((E2.shape[0], )); eta1=np.mean(h1)
for i in range(E2.shape[0]):
    e2=E2[i]; 
    a=-np.log((1-e1)/e2)
    b=-np.log(e1/(1-e2))
    # print(a, b)
    for l in range(20):
        perm=np.random.permutation(N)
        Y1=X1
        Y1=Y1[:,perm]
        #odbirci prve klase dolaze
        Sm=np.zeros((N,))
        k=0
        while (Sm[k]>a):
            i1=np.argmin(np.abs(x-Y1[0,k]))
            j1=np.argmin(np.abs(y-Y1[1,k]))
            Sm[k+1]=Sm[k]+h[i1,j1]
            k+=1
        k_mean[i]+=k-1
    k_mean[i]/=20   
    m1[i]=(a*(1-e1)+b*e1)/eta1

plt.figure()
plt.semilogx(E2, k_mean)
plt.semilogx(E2, m1)
plt.xlabel('$\epsilon_2$')
plt.legend(['$\hat{m}_1(\epsilon_2)$','$m_1(\epsilon_2)$'])

#%% Zavisnos broja potrebnih odbiraka od e1/e2 (za 2. klasu)
plt.close('all')
E1=np.logspace(-50, -0.01,  200)
E2=np.logspace(-50, -0.01,  200)

e2=1e-20
k_mean=np.zeros((E1.shape[0], ))
m2=np.zeros((E1.shape[0], )); eta2=np.mean(h2)
for i in range(E1.shape[0]):
    e1=E1[i]; 
    a=-np.log((1-e1)/e2)
    b=-np.log(e1/(1-e2))
    # print(a, b)
    for l in range(20):
        perm=np.random.permutation(N)
        Y1=X2
        Y1=Y1[:,perm]
        #odbirci prve klase dolaze
        Sm=np.zeros((N,))
        k=0
        while (Sm[k]<b):
            i1=np.argmin(np.abs(x-Y1[0,k]))
            j1=np.argmin(np.abs(y-Y1[1,k]))
            Sm[k+1]=Sm[k]+h[i1,j1]
            k+=1
        k_mean[i]+=k-1
    k_mean[i]/=20  
    m2[i]=(b*(1-e2)+a*e2)/eta2


plt.figure()
plt.semilogx(E1, k_mean)
plt.semilogx(E1, m2)
plt.xlabel('$\epsilon_1$')
plt.legend(['$\hat{m}_2(\epsilon_1)$','$m_2(\epsilon_1)$'])

e1=1e-20
k_mean=np.zeros((E2.shape[0], ))
m2=np.zeros((E2.shape[0], )); eta2=np.mean(h2)
for i in range(E2.shape[0]):
    e2=E2[i]; 
    a=-np.log((1-e1)/e2)
    b=-np.log(e1/(1-e2))
    # print(a, b)
    for l in range(20):
        perm=np.random.permutation(N)
        Y1=X2
        Y1=Y1[:,perm]
        #odbirci prve klase dolaze
        Sm=np.zeros((N,))
        k=0
        while (Sm[k]<b):
            i1=np.argmin(np.abs(x-Y1[0,k]))
            j1=np.argmin(np.abs(y-Y1[1,k]))
            Sm[k+1]=Sm[k]+h[i1,j1]
            k+=1
        k_mean[i]+=k-1
    k_mean[i]/=20   
    m2[i]=(b*(1-e2)+a*e2)/eta2

plt.figure()
plt.semilogx(E2, k_mean)
plt.semilogx(E2, m2)
plt.xlabel('$\epsilon_2$')
plt.legend(['$\hat{m}_2(\epsilon_2)$','$m_2(\epsilon_2)$'])

