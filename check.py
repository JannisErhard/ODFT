from math import exp, pi, sqrt, sin
import numpy as np
import matplotlib.pyplot as plt  
import scipy.integrate as integrate
import time

# if V(r) = a r^n then 2 <T> = n <V>

def H_den(x):
    return (exp(-abs(x)))**2/pi

def gaussian(alpha,x):
    # keep density normalized
    return (alpha/pi)**(3/2.)*exp(-alpha*x**2)


def rho(c,x):
    # density as expansion in basis set 
    p_norm = sum(c)
    p = 0
    for c_i,alpha in zip(c,basis_satz):
        p += c_i/p_norm*gaussian(alpha,x)
    return p


def kinetischer_energie_integrand(f,c,x):
    # (3 \pi^2)^{2/3} \frac{3}{10} \rho^{\frac{5}{3}} Yang-Parr p.108
    return (3*pi**2)**(2/3.)*3/10.*f(c,x)**(5/3.)

def kinetische_energie(c):
    # normal version
    res = integrate.quad(lambda x: 4*pi*x**2*kinetischer_energie_integrand(rho,c,x),0,np.inf)
    return res[0]

def gaussian_kernpotential_energie(c):
    # use boys function trick for coulomb energy
    V_eK=0
    for c_i,alpha in zip(c,basis_satz):
        V_eK += c_i*-1*sqrt(4.0*alpha/pi)
    return V_eK

def total_energy(c):
    norm = sum(c)
    c = [i/norm for i in c]
    E_kin = kinetische_energie(c) 
    E_c = gaussian_kernpotential_energie(c)
    E_tot = E_c + E_kin
    print(f"{E_c=} {E_kin=} {E_tot=}","Virial=",2*E_kin/E_c)
    print(integrate.quad(lambda x: 4*pi*x**2*rho(c,x),0,np.inf)[0])
    return E_tot


basis_set_size = 20
n = basis_set_size
basis_satz = [alpha for alpha in np.logspace(-1,8,n)]

basis_set_size = 100
n = basis_set_size
basis_satz = [1.7642452031482*2**i for i in np.linspace(-18,2,n)]
#basis_satz = [1.7642452031482/4.,1.7642452031482/2.,1.7642452031482,1.7642452031482*2,1.7642452031482*4]

c=[float(i) for i in range(1,n+1)]
norm = sum(c)
c_0 = [i/norm for i in c]



# checking if rho is defined properly 
result = integrate.quad(lambda x: 4*pi*x**2*rho(c,x),0,np.inf)

# checking if functions are defined properly 
result = integrate.quad(lambda x: 4*pi*x**2*(basis_satz[1]/pi)**(3./2.)*exp(-basis_satz[1]*x**2),0,np.inf)

# E_{TF}[\rho] = 4 \pi \int_0^{\infty} dr r**2 e_{TF}[\rho] 
V_TF =  kinetische_energie(c)




## plot density and kinetic energy functional
plot = False 
if plot:
    r = [float(i/19) for i in range(-190,190)]
    values = get_values(r,c)
    EK_plot = []
    for x in r:
        EK_plot.append(kinetischer_energie_integrand(c,x))
    def get_values(r,c):
        values = []
        for x in r:
            values.append(rho(c,x))
        return values 
    plt.plot(r,values)
    plt.plot(r,EK_plot)
    plt.show()

spikyness = [i for i in np.linspace(1,-4,1000)] 


# run through a number of basis sets, desribing very dense to very disperse charge blobs for Hydrogen, to plot sharpness against energy, to visualize the minimum 
spike_test=False
if spike_test:
    for spike in spikyness:
        basis_satz = [alpha for alpha in np.logspace(spike+1.5,spike-1.5,19)]
        V_Ke = gaussian_kernpotential_energie(c)
        V_TF = kinetische_energie(c)
        print(spike, spike+8, V_Ke, V_TF, )

l_1D_test=False
if l_1D_test:
    spikyness = [i for i in np.logspace(-4,4,1000)] 
    #  just one function, steepness test 
    for spike in spikyness:
        basis_satz = [spike]
        c = [1.0]
        V_Ke = gaussian_kernpotential_energie(c)
        V_TF = kinetische_energie(c)
        print(spike,V_Ke, V_TF, total_energy(c))


# optimise the basis set coeficients to find the energy minimum for Hydrogen
from scipy.optimize import minimize
l_Optimize = True
plot = True 

if l_Optimize:
    c0 = np.array(c)
    print(total_energy(c), total_energy(c0))
    bnds = ((0, 1.0),)*n # only use positive numbers since there can not be any negative region in the density
    res = minimize(total_energy,c,bounds=bnds)
    print(total_energy(res.x))
    
    norm_final = sum(res.x)
    c_f = [i/norm_final for i in res.x]
    
    if True:
        for i,j in zip(c_f,basis_satz):
            print(i,j)

print(integrate.quad(lambda x: 4*pi*x**2*H_den(x),0,np.inf))
print(integrate.quad(lambda x: 4*pi*x**2*rho(res.x,x),0,np.inf))

if plot:
    r = [float(i/19) for i in range(-190,190)]
    rho_initial = [rho(c_0,x) for x in r ]
    rho_final = [rho(c_f,x) for x in r ]
    Reference = [H_den(x) for x in r ]
    plt.plot(r,rho_initial)
    plt.plot(r,rho_final)
    plt.plot(r,Reference)
    plt.show()
