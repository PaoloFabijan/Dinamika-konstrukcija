from tkinter import X
import numpy as np
from scipy.linalg import eigh
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sympy import N

b = 0.003
h = 0.04
I = (b**3*h)/12
E = 8.6E10

K = np.array([[1/(729/(15625*E*I)), 1/(3344/(46875*E*I)), 1/(1429/(46875*E*I))], 
              [1/(3344/(46875*E*I)), 1/(2304/(15625*E*I)), 1/(3344/(46875*E*I))], 
              [1/(1429/(46875*E*I)), 1/(3344/(46875*E*I)), 1/(729/(15625*E*I))]])

M = np.array([[4/9.81, 0, 0], 
              [0, 2/9.81, 0], 
              [0, 0, 4/9.81]])

w, v = eigh(K, M)

omega = np.sqrt(w)
freq = omega/np.pi

phi_1 = v[:,0]
phi_2 = v[:,1]
phi_3 = v[:,2]

# Rjesavanje jednadzbe kretanja

t_end = 3
dt = 0.001
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
x = np.zeros([3,n])
v = np.zeros([3,n])
a = np.zeros([3,n])

#pocetni uvjeti:
x[:,0] = phi_2
a[:,0] = np.dot(inv(M), np.dot(K, x[:,0]))

#efektivna matrica krutosti (konstantna je):
K_eff = b1*M + K
K_eff_inv = inv(K_eff)

#petlja kojom koracamo u vremenu i racunamo nepoznate pomake, brzine i ubrzanja i svakom trenutku:
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
plt.plot(t, x[0,:], "r")
plt.plot(t, x[1,:], "g")
plt.plot(t, x[2,:], "b")
plt.plot([0, t[-1]],[0, 0], 'k--')
plt.legend(['masa 1', 'masa 2', 'masa 3'], loc ="lower right")
plt.xlabel('Vrijeme t (s)')
plt.ylabel('Pomak x (mm)')


# Crtanje pomaka masa na gredi
# loc = 0
# L1 = 0.32
# L2 = 0.64
# dataX = np.array([0, L1, L1+L2, L1+2*L2, 2*L1+2*L2])
# dataY = np.array([0, x[0,loc], x[1,loc], x[2,loc], 0])
# plt.plot(dataX, dataY, "g")
# plt.plot([0, 2*L1+2*L2], [0, 0], "k--")

# plt.plot(L1, x[0,loc], "ro")
# plt.plot(L1+L2, x[1,loc], "ro")
# plt.plot(L1+2*L2, x[2,loc], "ro")


# plt.show()

print(omega)