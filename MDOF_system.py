import numpy as np
from scipy.linalg import eigh
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sympy import N
from stiffness_matrix import K 
from mass_matrix import M 

# Obrisani su dijelovi matrice krutosti i masa koji su povezani sa pomacima oslonaca
M = np.delete(M, [0, -2], 0)
M = np.delete(M, [0, -2], 1)

K = np.delete(K, [0, -2], 0)
K = np.delete(K, [0, -2], 1)

w, v = eigh(K, M)

omega = np.sqrt(w)
freq = omega/np.pi

phi_1 = v[:,1]
phi_2 = v[:,3]
phi_3 = v[:,5]

# Rjesavanje jednadzbe kretanja

t_end = 10
dt = 0.1
n = int(t_end/dt)

# Newmarkovi parametri
delta = 1/2 
alpha = 1/4
b1 = 1/(alpha*dt**2)
b2 = -1/(alpha*dt)
b3 = 1- 1/(2*alpha)
b4 = delta/(alpha*dt)
b5 = 1- delta/alpha
b6 = (1 - delta/(2*alpha))*dt

t = np.zeros(n)
x = np.zeros([8,n])
v = np.zeros([8,n])
a = np.zeros([8,n])

#pocetni uvjeti:
x[:,0] = phi_1
a[:,0] = np.dot(inv(M), np.dot(K, x[:,0]))

#efektivna matrica krutosti (konstantna je):
K_eff = b1*M + K
K_eff_inv = inv(K_eff)

#petlja kojom koracamo u vremenu i racunamo nepoznate pomake, brzine i ubrzanja u svakom trenutku:
for i in range(n-1):
    t[i+1] = (i+1)*dt
    
    #efektivni vektor opterecenja (mijenja se u svakom trenutku):
    F_eff = np.dot(M, (b1*x[:,i] - b2*v[:,i] - b3*a[:,i]))
    
    #pomaci dobiveni iz diskretizirane jednadzbe kretanja s Newmarkovim izrazima za ubrzanje:
    x[:,i+1] = np.dot(K_eff_inv, F_eff)
    
    #brzine i ubrzanja iz Newmarkovih izraza:
    v[:,i+1] = b4*(x[:,i+1] - x[:,i]) + b5*v[:,i] + b6*a[:,i]
    a[:,i+1] = b1*(x[:,i+1] - x[:,i]) + b2*v[:,i] + b3*a[:,i]

# CRTANJE GRAFOVA    
plt.plot(t, x[1,:], "r")
plt.plot(t, x[3,:], "g")
plt.plot(t, x[5,:], "b")
plt.plot([0, t[-1]],[0, 0], 'k--')
plt.legend(['masa 1', 'masa 2', 'masa 3'], loc ="lower right")
plt.xlabel('Vrijeme t (s)')
plt.ylabel('Pomak x (mm)')
# plt.xlim(8, 10)


# Crtanje pomaka masa na gredi
# loc = 0
# L1 = 0.32
# L2 = 0.64
# dataX = np.array([0, L1, L1+L2, L1+2*L2, 2*L1+2*L2])
# dataY = np.array([0, x[1,loc], x[3,loc], x[5,loc], 0])
# plt.plot(dataX, dataY, "g")
# plt.plot([0, 2*L1+2*L2], [0, 0], "k--")

# plt.plot(L1, x[1,loc], "ro")
# plt.plot(L1+L2, x[3,loc], "ro")
# plt.plot(L1+2*L2, x[5,loc], "ro")

# plt.show()
print(omega)