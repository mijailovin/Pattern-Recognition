import numpy as np
import scipy
from scipy import io
import matplotlib.pyplot as plt

from skimage import io
from skimage import util

from numpy.linalg import *
%matplotlib qt
# %matplotlib inline

from skimage import exposure, color, filters, util, io

plt.close('all')
#%% Generisnje podataka 1
N=500
n=2
# np.random.seed(5)
M1=np.array([[1],[2]])
S1=np.array([[0.3, 0],\
              [0, 0.1]])
M2=np.array([[2],[6]])
S2=np.array([[2, -0.5],\
              [-0.5, 1]])

X1=np.zeros((2,N))
X2=np.zeros((2,N))
for i in range(N):
    X1[:,i]=np.random.multivariate_normal(M1.flatten(), S1).T
    X2[:,i]=np.random.multivariate_normal(M2.flatten(), S2).T

plt.figure()
plt.plot(X1[0,:],X1[1,:],'x')
plt.plot(X2[0,:],X2[1,:],'x')
plt.legend(['$K_1$','$K_2$'])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')

#%% Generisanje podataka 2
N=500
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
#%% Generisanje podataka 3
N=500
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
#%% 1. Numericka metoda
from scipy.integrate import quad

def f(x, P):
    return  (P/(2*np.pi)**0.5)*np.exp(-0.5*x**2)

def num_metod_1(K1, K2):
    N1=K1.shape[1]
    N2=K2.shape[1]
    
    P1=N1/(N1+N2)
    P2=N2/(N1+N2)
    M1=np.reshape(np.mean(K1, axis=1), (2,1))
    M2=np.reshape(np.mean(K2, axis=1), (2,1))
    S1=np.cov(K1)
    S2=np.cov(K2)
    
    
    ds=0.01
    s=np.arange(0, 1, ds)
    epsilon=np.zeros_like(s)
    epsilon_min=1e6
    s_opt=-1
    V_opt=0
    v0_opt=0
    for i in range(s.shape[0]):
        V=inv((s[i]*S1+(1-s[i])*S2)) @ (M2-M1)
        sigma12=V.T@S1@V
        sigma22=V.T@S2@V
        v0=(s[i]*sigma12*V.T@M2+(1-s[i])*sigma22*V.T@M1) / (s[i]*sigma12+(1-s[i])*sigma22)
        # print(v0)
        eta1=V.T@M1+v0
        eta2=V.T@M2+v0
        
        
        
        e1 = quad(f, -eta1/sigma12**0.5, np.inf, args=(P1))[0]
        e2 = quad(f, -np.inf, -eta2/sigma22**0.5, args=(P2))[0]
        epsilon[i]=e1+e2
        
        if (epsilon[i]<epsilon_min):
            epsilon_min=epsilon[i]
            s_opt=s[i]
            V_opt=V
            v0_opt=v0
            
    plt.figure()
    plt.plot(s, epsilon)
    plt.show()
    
    return s_opt, V_opt, v0_opt
#%% Linearni klasifikator- 1. Numericka metoda
plt.close('all')

s_opt, V, v0= num_metod_1(X1, X2)
x1=np.arange(0,8,0.1)
x2=(-(v0+V[0]*x1)/V[1]).flatten()

  

plt.figure()
plt.plot(X1[0,:],X1[1,:],'x')
plt.plot(X2[0,:],X2[1,:],'x')
plt.plot(x1, x2)
plt.axis('equal')

#%% Linearni klasifikator- 2/3. Numericka metoda
from klasifikator_linerni import *
plt.close('all')
s_opt, v0, Neps, M1, M2, S1, S2=num_metod_3(X1, X2)   
V=inv((s_opt*S1+(1-s_opt)*S2))@(M2-M1)
# print(s_opt, '\n', V,'\n', v0)
#%%
x1=np.arange(-4, 10, 0.1)
x2=(-(v0+V[0]*x1)/V[1]).flatten()

plt.figure()
plt.plot(X1[0,:],X1[1,:],'x')
plt.plot(X2[0,:],X2[1,:],'x')
plt.plot(x1, x2)
plt.axis('equal')
plt.legend(['$K_1$','$K_2$'])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

#%% Metoda zeljenog izlaza- linearni

# Gama=np.ones((2*N,1))
Gama=np.concatenate((np.ones((N,1)), 2*np.ones((N,1))), axis=0)
Gama=np.concatenate((2*np.ones((N,1)), np.ones((N,1))), axis=0)

U=np.concatenate( (np.concatenate((-np.ones((1, N)), np.ones((1, N))), axis=1), \
                   np.concatenate((-X1, X2), axis=1)), axis=0)


W=inv((U@U.T))@U@Gama

v0=W[0]
V=W[1:]
x1=np.arange(-4, 8, 0.1)
x2=(-(v0+V[0]*x1)/V[1]).flatten()

print(V, v0)
plt.figure()
plt.plot(X1[0,:],X1[1,:],'x')
plt.plot(X2[0,:],X2[1,:],'x')
plt.plot(x1, x2)
plt.legend(['$K_1$','$K_2$', 'pocetan slučaj', '$\omega_2$ bitnije', '$\omega_1$ bitnije'])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')

#%% Kvadratni klasifikator- metodom zeljenog izlaza

Gama=np.ones((2*N,1))
U=np.concatenate( (np.concatenate((-np.ones((1, N)), np.ones((1, N))), axis=1),  \
                   np.concatenate((-X1,X2), axis=1), \
                   np.concatenate((-X1[0:1,:]**2, X2[0:1,:]**2), axis=1), \
                   np.concatenate((-2*X1[0:1,:]*X1[1:2,:], 2*X2[0:1,:]*X2[1:2,:]), axis=1),\
                   np.concatenate((-X1[1:2,:]**2, X2[1:2,:]**2), axis=1)), axis=0) 

W=inv((U@U.T)) @ U @ Gama

# W=np.array([[-2.1791],[0.3733],[0.3544],[-0.0097],[-0.0110],[-0.0086]])
print(W)

v0=W[0]; V=W[1:3]; 
Q=np.array([[W[3,0], W[4,0]],[W[4,0], W[5,0]]])


# iscrtavanje 
plt.close('all')
Ng=200
x1g=np.linspace(min(np.min(X1[0,:]), np.min(X2[0,:])), max(np.max(X1[0,:]), np.max(X2[0,:])), Ng)
x2g=np.linspace(min(np.min(X1[1,:]), np.min(X2[1,:])), max(np.max(X1[1,:]), np.max(X2[1,:])), Ng)


h=np.zeros((Ng,Ng)) #vrednosti diskriminacione f-je
for i in range(0,Ng):
    for j in range(0,Ng):
        Xi=np.array([[x1g[i]],[x2g[j]]])
        # h[i,j]=x1g[i]**2*Q[0,0]+2*x1g[i]*x2g[j]*Q[0,1]+x2g[j]**2*Q[1,1]+x1g[i]*V[0]+x2g[j]*V[1]+v0
        h[i,j]=Xi.T@Q@Xi + V.T@Xi + v0

plt.figure()
plt.plot(X1[0,:], X1[1,:],'x')
plt.plot(X2[0,:], X2[1,:],'x')
plt.xlabel('$x_1$'); plt.ylabel('$x_2$');
# plt.title('Odbirci klasa')
plt.legend(['K1','K2'])
plt.contour(x1g, x2g, h.T, levels=[0])
plt.axis('equal')
plt.show()

#%% Klasifikacija na nekom novom odbirku

Xi=np.array([[2],[4]])
out=Xi.T@Q@Xi + V.T@Xi + v0

print(out)