from math import exp, pi, sqrt, sin
import numpy as np
import matplotlib.pyplot as plt  
import scipy.integrate as integrate
import time


def gaussian(alpha,x):
    # keep density normalized
    return (alpha/pi)**(3/2.)*exp(-alpha*x**2)


def rho(c,x):
    # density as expansion in basis set 
    # keep density normalized 
    p_norm = sum(c)
    p = 0
    for i,c_i in enumerate(c):
        p += c_i/p_norm*(basis_satz[i]/pi)**(3/2.)*exp(-basis_satz[i]*x**2)
    return p


def kinetischer_energie_integrand(f,c,x):
    # (3 \pi^2)^{3/2} \frac{3}{10} \rho^{\frac{5}{3}} Yang-Parr p.108
    return (3*pi**2)**(2/3.)*3/10.*f(c,x)**(5/3.)

def kinetische_energie(c):
    # normal version
    res = integrate.quad(lambda x: 4*pi*x**2*kinetischer_energie_integrand(rho,c,x),0,np.inf)
    #print(f"{c}, {res[0]}")
    #time.sleep(0.4)
    return res[0]

def gaussian_kernpotential_energie(c):
    # use boys function trick for coulomb energy
    V_eK=0
    for i,alpha in enumerate(basis_satz):
        V_eK += c[i]*-1*sqrt(4.0*alpha/pi)
    return V_eK

def total_energy(c):
    norm = sum(c)
    c = [i/norm for i in c]
    E_tot = gaussian_kernpotential_energie(c)+kinetische_energie(c)  
    print(E_tot)
    return E_tot


basis_set_size = 10
n = basis_set_size
basis_satz = [alpha for alpha in np.logspace(-4,4,n)]
c=[float(i) for i in range(1,n+1)]
norm = sum(c)
c_0 = [i/norm for i in c]



# checking if rho is defined properly 
result = integrate.quad(lambda x: 4*pi*x**2*rho(c,x),0,np.inf)

print("density integral", result)

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
        print(spike, spike+8, V_Ke, V_TF[0])

# optimise the basis set coeficients to find the energy minimum for Hydrogen
from scipy.optimize import minimize
c0 = np.array(c)
print(total_energy(c), total_energy(c0))
bnds = ((0, 1.0),)*n
res = minimize(total_energy,c,bounds=bnds)
print(res.x)
print(total_energy(res.x))

norm_final = sum(res.x)
c_f = [i/norm_final for i in res.x]


plot = True 
if plot:
    r = [float(i/190) for i in range(-190,190)]
    rho_initial = [rho(c_0,x) for x in r ]
    rho_final = [rho(c_f,x) for x in r ]
    #plt.plot(r,rho_initial)
    plt.plot(r,rho_final)
    plt.show()
