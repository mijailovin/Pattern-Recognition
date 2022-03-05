import numpy as np
import scipy
from scipy import io
import matplotlib.pyplot as plt

from skimage import io
from skimage import util
from skimage import data, draw, io

from numpy.linalg import *
%matplotlib qt
# %matplotlib inline

from skimage import exposure, color, filters, util, io
from skimage import feature
from skimage.morphology import disk, dilation
from skimage.transform import resize

plt.close('all')
#%%
img=util.img_as_float(io.imread('./BazaSlova/bazaA001.bmp'))
plt.figure()
plt.imshow(img, cmap='gray', vmin=0, vmax=1);

img=util.img_as_float(io.imread('./BazaSlova/bazaA008.bmp'))
plt.figure()
plt.imshow(img, cmap='gray', vmin=0, vmax=1);


#%% Sklanjanje granica sa slike, da ostanu samo slova
#prvo se uklanja crni border

def crop_dim(img):
    '''
    samo racuna dimenzije nove slike
    '''
    [nr, nc]=img.shape
    #isecanje crnog okvira
    
    # poc=0     #krece odozgo
    # while (poc<nr-1 and  np.sum(img[poc,:])/nc<0.8 ): 
    #     poc+=1
    poc=40
    kraj=nr-30
    levo=20
    desno=nc-20
        
    # kraj=nr-1
    # while (kraj>0 and np.sum(img[kraj,:])/nc<0.8 ):
    #     kraj=kraj-1
        
    # levo=0
    # while (levo<nc-1 and  np.sum(img[:,levo])/nr<0.8 ):
    #     levo=levo+1
    
    # desno=nc-1
    # while (desno>0 and  np.sum(img[:,desno])/nr<0.8 ):
    #     desno-=1
        
    # print(poc, kraj, levo, desno)
    #sada canny
    edges = feature.canny(img, sigma=3)
    # plt.figure()
    # plt.imshow(edges[poc:kraj, levo:desno], cmap='gray');
    # plt.show()
    
    # figure, ax = plt.subplots(1,2)
    # ax[0].imshow(img[poc:kraj, levo:desno], cmap='gray', vmin=0, vmax=1);
    # ax[0].set_title('Ukljonjene crne ivice')
    # ax[1].imshow(edges[poc:kraj, levo:desno], cmap='gray', vmin=0, vmax=1);
    # ax[1].set_title('Detekcija ivica')
    # plt.show()
    #sada dodatno isecanje
    while (poc<nr-1 and  np.max(edges[poc, levo+1:desno-1])==0 ): 
        poc+=1
    while (kraj>0 and np.max(edges[kraj, levo+1:desno-1])==0 ):
        kraj=kraj-1
    while (levo<nc-1 and  np.max(edges[poc+1:kraj-1, levo])==0 ):
        levo=levo+1
    while (desno>0 and  np.max(edges[poc+1:kraj-1, desno])==0 ):
        desno-=1
    
    return poc, kraj, levo, desno
    
#%% Predobrada slika

samoglasnici=['A', 'O', 'U', 'E', 'I']
N=120

# da se uradi: slova jedna preko drugog
img_all=np.zeros((200,100)) 
j=4
for i in range(0,N):
    print('a', end='')
    ime='./BazaSlova/baza'+samoglasnici[j]+str('{:0{}}'.format(i+1, 3))+'.bmp'
    img=util.img_as_float(io.imread(ime))
    poc, kraj, levo, desno =crop_dim(img)
    img_new=resize(img[poc:kraj, levo:desno], (200,100))
    img_all=img_all + img_new
    
    # figure, ax = plt.subplots(1,2)
    # ax[0].imshow(img, cmap='gray', vmin=0, vmax=1);
    # ax[1].imshow(img[poc:kraj, levo:desno], cmap='gray', vmin=0, vmax=1);
    # plt.show()
plt.figure()
plt.imshow(img_all, cmap='gray')
#%% Binarizacija slike
#promenljivim pragom
plt.close('all')
from skimage.filters import threshold_otsu

#priloziti slike za svako slovo
img=util.img_as_float(io.imread('./BazaSlova/bazaI001.bmp'))
poc, kraj, levo, desno=crop_dim(img)
img_new=resize(img[poc:kraj, levo:desno], (200, 100))
thr=threshold_otsu(img_new)

plt.figure()
plt.hist(img[poc:kraj, levo:desno].flatten(), bins=100, density=True)
plt.vlines(thr, 0, 70, 'green','--')
plt.legend(['Otsu', 'histogram'])
plt.title('Histogram isečene slike')
plt.show()

print(thr)

figure, ax = plt.subplots(1,3, figsize=(12,6))
ax[0].imshow(img, cmap='gray', vmin=0, vmax=1); ax[0].set_title('Originalna slika')
ax[1].imshow(img_new, cmap='gray', vmin=0, vmax=1); ax[1].set_title('Isečena slika')
ax[2].imshow(img_new>thr, cmap='gray', vmin=0, vmax=1); ax[2].set_title('Binarizovana slika')
plt.show()
 
#%% Feautures
def img_final(img):
    poc, kraj, levo, desno=crop_dim(img)
    img_new=resize(img[poc:kraj, levo:desno], (200, 100))
    thr=threshold_otsu(img_new)
    img_bin=(img_new>thr)*1.0
    
    return img_bin
def features(img):
    '''
    img: dim (200, 100)
    '''
    X=np.zeros((5,1))
    X[0]=np.mean(img[80:120, 40:60]) #centralni deo slike
    X[1]=np.mean(img[175:, 30:70]) # donji- centralni deo slike
    # X[1]=np.mean(img[175:, 30:70])-np.mean(img[:25, 30:70])
    X[2]=np.mean(img[75:125, 0:20]) #levi- srednji deo slike
    # X[3]=np.mean(img[0:20, 25:75]) # gornji- centralni deo slike
    X[3]=np.mean(img[30:70, 0:25])-np.mean(img[120:170, 0:25]) # razlika delova sa leve strane
    X[4]=np.mean(img[:, 75:]) # desni deo slike
    return X
#%% Iscrtavanje feature-a
plt.close('all')
# img=np.ones((200,100,3))
# row, col = draw.rectangle_perimeter(start=(80,40), end=(120, 60))
# img[row, col] = [0, 0, 1] 
# plt.figure()
# plt.imshow(img)

# img=np.ones((200,100,3))
# row, col = draw.rectangle_perimeter(start=(175,30), end=(198, 70))
# img[row, col] = [0, 0, 1] 
# plt.figure()
# plt.imshow(img)

# img=np.ones((200,100,3))
# row, col = draw.rectangle_perimeter(start=(75,1), end=(125, 20))
# img[row, col] = [0, 0, 1] 
# plt.figure()
# plt.imshow(img)

# img=np.ones((200,100,3))
# row, col = draw.rectangle_perimeter(start=(30,1), end=(70, 25))
# img[row, col] = [0, 0, 1] 
# row, col = draw.rectangle_perimeter(start=(120,1), end=(170, 25))
# img[row, col] = [1, 0, 0] 
# plt.figure()
# plt.imshow(img)

img=np.ones((200,100,3))
row, col = draw.rectangle_perimeter(start=(2,75), end=(198, 98))
img[row, col] = [0, 0, 1] 
plt.figure()
plt.imshow(img)

#%% Histogram feature-a
j=1
plt.figure()
n=2
N=120
samoglasnici=['A', 'O', 'U', 'E', 'I']
X=np.zeros((n,N))
for i in range(0,N):
    print(samoglasnici[j], end='')
    ime='./BazaSlova/baza'+samoglasnici[j]+str('{:0{}}'.format(i+1, 3))+'.bmp'
    img=util.img_as_float(io.imread(ime))
    img_bin=img_final(img)
    
    X[:,i:i+1]=features(img_bin)[1:n,:]
plt.hist(X[0,:], bins=10)
plt.title(samoglasnici[j])
plt.show()

#%% Izvlacenje obelezja 

#(96/120- train set)
np.random.seed(15)
n=5 # broj karakterisitka (feature-a)

X_train=[]
X_test=[]
N_train=round(120*0.8); print(N_train)
for i in range(len(samoglasnici)):
    Xi=np.zeros((n,N))
    for j in range(N):
        print(samoglasnici[i], end='')
        ime='./BazaSlova/baza'+samoglasnici[i]+str('{:0{}}'.format(j+1, 3))+'.bmp'
        img=util.img_as_float(io.imread(ime))
        img_bin=img_final(img)
        
        Xi[:,j:j+1]=features(img_bin)[0:n,:]
    perm=np.random.permutation(N)
    Xi=Xi[:, perm]
    X_train.append(Xi[:,:N_train])
    X_test.append(Xi[:,N_train:])

#%% Crtanje

fig = plt.figure()
ax = plt.axes(projection='3d')
for i in range(len(samoglasnici)):
    ax.scatter(X_train[i][0,:], X_train[i][1,:], X_train[i][2,:])
plt.legend(samoglasnici)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.set_title('Prikaz odbiraka dimenzije 3x1')
plt.show()

#%% Projektovanje klasifikatora
M=[]
S=[]
for i in range(len(samoglasnici)):
    M.append(np.reshape(np.mean(X_train[i],axis=1), (n,1)))
    S.append(np.cov(X_train[i]))

#pretpostavlja se Gausova raspodela


# Testiranje klasifikatora
import seaborn as sn

L=len(samoglasnici)
cm=np.zeros((L, L))
pred=[]
for i in range(L):
    T=X_test[i]
    pred.append(np.zeros((N-N_train, ), dtype=int))
    #sada idu odbirci samo jedne klase za test
    for j in range(N-N_train):
        Xj=T[:,j:j+1]
        f=np.zeros((L, ))
        for k in range(L):
            f[k]=1/(2*np.pi*det(S[0])**0.5)*np.exp( -0.5*(Xj-M[k]).T@inv(S[k])@(Xj-M[k]) )

        ind=np.argmax(f)
        pred[i][j]=ind
        cm[ind,i]+=1
# print(cm)

plt.figure()
sn.heatmap(cm, annot=True, xticklabels=samoglasnici, yticklabels=samoglasnici, cmap='Blues')
  
print(np.trace(cm)/np.sum(cm))

#%% Primeri pravilnih/nepravilnih klasifikacija
plt.close('all')
for i in range(L):
    P=np.where(pred[i]!=i)[0]
    print(P)
    for p in P: 
        ime='./BazaSlova/baza'+samoglasnici[i]+str('{:0{}}'.format(p+1, 3))+'.bmp'
        img=util.img_as_float(io.imread(ime))        
        figure, ax = plt.subplots(1,2, figsize=(8, 5))
        ax[0].imshow(img, cmap='gray', vmin=0, vmax=1);
        ax[1].imshow(img_bin, cmap='gray', vmin=0, vmax=1);
        plt.suptitle(samoglasnici[i]+' klasifikovano kao: '+samoglasnici[pred[i][p]])
        plt.show()
#%% Primer pravilnih klasifikacija
plt.close('all')
for i in range(L):
    ime='./BazaSlova/baza'+samoglasnici[i]+str('{:0{}}'.format(2+1, 3))+'.bmp'
    img=util.img_as_float(io.imread(ime))
    img_bin=img_final(img)
    
    figure, ax = plt.subplots(1,2, figsize=(8, 5))
    ax[0].imshow(img, cmap='gray', vmin=0, vmax=1);
    ax[1].imshow(img_bin, cmap='gray', vmin=0, vmax=1);
    plt.suptitle('Tačna klasifikacija slova: '+samoglasnici[i])
    plt.show()