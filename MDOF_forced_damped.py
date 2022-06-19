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

phi_1 = v[:,0]
phi_2 = v[:,1]
phi_3 = v[:,2]

# Koeficijent prigušenja
zeta = 0.045611
a0 = zeta * ((2*omega[0] * omega[1])/(omega[0] + omega[1]))
a1 = zeta * (2/(omega[0] + omega[1]))

# Matrica prigušenja
C = a0*M + a1*K

# Ubrzanja i pomaci stola B
data = np.genfromtxt("Ubrzanja.txt", names = ("TB_acc"))
TB_acc = data["TB_acc"]

# a_g = np.zeros([len(data), 1])
a_g = TB_acc * 0.001
a_g = a_g.transpose() # Ubrzanja stola B
x_g = (a_g * 0.01**2) # Pomaci stola B

# Import podataka iz ispitivanja (test 3)
exp_data = np.genfromtxt("test3.csv", delimiter=",", names=("Time", "M1", "M2", "M3", "TA", "TB"))
m1 = exp_data["M1"] * 0.001
m2 = exp_data["M2"] * 0.001
m3 = exp_data["M3"] * 0.001
ta = exp_data["TA"] * 0.001
tb = exp_data["TB"] * 0.001

# Rjesavanje jednadzbe kretanja
t_end = 39.81
dt = 0.01
n = 3982
# n = int(t_end/dt)

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

# Pocetni uvjeti:
x[1,0] = m1[0] # Pomak mase 1
x[3,0] = m2[0] # Pomak mase 2
x[5,0] = m3[0] # Pomak mase 3
a[:,0] = np.dot(inv(M), (np.dot(-C, v[:,0]) + np.dot(-K, x[:,0])))

# Efektivna matrica krutosti (konstantna je):
K_eff = b1*M + K + b4*C
K_eff_inv = inv(K_eff)

# Petlja kojom koracamo u vremenu i racunamo nepoznate pomake, brzine i ubrzanja u svakom trenutku:
for i in range(n-1):
    t[i+1] = (i+1)*dt

    # Vektor vanjskog opterećenja
    e = (np.array([0, 1/6, 0, 1/2, 0, 5/6, 0, 0])).transpose() * x_g[i]
    F = np.dot((np.dot(-M, e)), a_g[i])
    
    # Efektivni vektor opterecenja (mijenja se u svakom trenutku):
    F_eff = F + np.dot(M, (b1*x[:,i] - b2*v[:,i] - b3*a[:,i])) + np.dot(C, (b4*x[:,i] - b5*v[:,i] - b6*a[:,i]))
    
    # Pomaci dobiveni iz diskretizirane jednadzbe kretanja s Newmarkovim izrazima za ubrzanje:
    x[:,i+1] = np.dot(K_eff_inv, F_eff)
    
    # Brzine i ubrzanja iz Newmarkovih izraza:
    v[:,i+1] = b4*(x[:,i+1] - x[:,i]) + b5*v[:,i] + b6*a[:,i]
    a[:,i+1] = b1*(x[:,i+1] - x[:,i]) + b2*v[:,i] + b3*a[:,i]

# CRTANJE GRAFOVA  
# plt.plot(t, m1*1000, "r") # Dijagram pomaka za masu 1 (Eksperiment)
plt.plot(t, m2*1000, "r") # Dijagram pomaka za masu 2 (Eksperiment)
# plt.plot(t, m3*1000, "r") # Dijagram pomaka za masu 3 (Eksperiment)

# plt.plot(t, -x[1,:]*1000, "g") # Dijagram pomaka za masu 1 (Python)
plt.plot(t, -x[3,:]*1000, "g") # Dijagram pomaka za masu 2 (Python)
# plt.plot(t, -x[5,:]*1000, "b") # Dijagram pomaka za masu 3 (Python)
plt.plot([0, t[-1]],[0, 0], 'k--')
plt.legend(['masa 1', 'masa 2', 'masa 3'], loc ="lower right")
plt.xlabel('Vrijeme t (s)')
plt.ylabel('Pomak x (mm)')

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

plt.show()

# print(np.dot(K_eff_inv, F_eff))