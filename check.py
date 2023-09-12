from math import exp, pi, sqrt, sin
import numpy as np
import matplotlib.pyplot as plt  
import scipy.integrate as integrate


basis_satz = [alpha for alpha in np.logspace(-4,4,19)]

c=[float(i) for i in range(1,20)]

def den_norm(c):
    return sum([i for i in c])

deniy =  den_norm(c)
for i,c_i in enumerate(c):
   c[i] = c[i]/deniy


def rho(c,x):
    # density as expansion in basis set 
    # keep density normalized 
    p_norm = den_norm(c)
    p = 0
    for i,c_i in enumerate(c):
        p += c_i/p_norm*(basis_satz[i]/pi)**(3/2.)*exp(-basis_satz[i]*x**2)
    return p


def kinetischer_energie_integrand(c,x):
    # (3 \pi^2)^{3/2} \frac{3}{10} \rho^{\frac{5}{3}} Yang-Parr p.108
    return (3*pi**2)**(2/3.)*3/10.*rho(c,x)**(5/3.)

def kinetische_energie(c):
    res= integrate.quad(lambda x: 4*pi*x**2*kinetischer_energie_integrand(c,x),0,np.inf)
    print(c,res)
    return res

def gaussian_kernpotential_energie(c):
    # use boys function trick for coulomb energy
    V_eK=0
    for i,alpha in enumerate(basis_satz):
        V_eK += c[i]*-1*sqrt(4.0*alpha/pi)
    return V_eK

def total_energy(c):
    return gaussian_kernpotential_energie(c)+kinetische_energie(c)[0]




# checking if rho is defined properly 
result = integrate.quad(lambda x: 4*pi*x**2*rho(c,x),0,np.inf)

# checking if functions are defined properly 
result = integrate.quad(lambda x: 4*pi*x**2*(basis_satz[1]/pi)**(3./2.)*exp(-basis_satz[1]*x**2),0,np.inf)

# E_{TF}[\rho] = 4 \pi \int_0^{\infty} dr r**2 e_{TF}[\rho] 
V_TF = integrate.quad(lambda x: 4*pi*x**2*kinetischer_energie_integrand(c,x),0,np.inf)




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
        #basis_satz = [alpha for alpha in np.logspace(spike+2,spike-2,19)]
        basis_satz = [alpha for alpha in np.logspace(spike+1.5,spike-1.5,19)]
        V_Ke = gaussian_kernpotential_energie(c)
        V_TF = kinetische_energie(c)
        #integrate.quad(lambda x: 4*pi*x**2*kinetischer_energie_integrand(c,x),0,np.inf)
        print(spike, spike+8, V_Ke, V_TF[0])

# optimise the basis set coeficients to find the energy minimum for Hydrogen
from scipy.optimize import minimize
c0 = np.array(c)
print(total_energy(c), total_energy(c0))
res = minimize(total_energy,c)
print(res.x)
print(total_energy(res.x))

