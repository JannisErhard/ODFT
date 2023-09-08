from math import exp, pi, sqrt, sin
import numpy as np
import matplotlib.pyplot as plt  
import scipy.integrate as integrate


basis_satz = [alpha for alpha in np.logspace(-4,4,19)]

c=[float(i) for i in range(1,19)]

def den_norm(c):
    return sum([i for i in c])

deniy =  den_norm(c)
for i,c_i in enumerate(c):
   c[i] = c[i]/deniy


def rho(c,x):
    # density as expansion in basis set 
    p = 0 
    for i,c_i in enumerate(c):
        p += c_i*(basis_satz[i]/pi)**(3/2.)*exp(-basis_satz[i]*x**2)
    return p


def kinetische_energie(rho,c,x):
    # (3 \pi^2)^{3/2} \frac{3}{10} \rho^{\frac{5}{3}}
    return (3*pi**2)**(3/2.)*3/10.*rho(c,x)**(5/3.)


def gaussian_kernpotential_energie(c):
    # use boys function trick for coulomb energy
    V_eK =0
    for alpha in  basis_satz:
        V_eK += -1*sqrt(4.0*alpha/pi)
    return V_eK





# checking if rho is defined properly 
result = integrate.quad(lambda x: 4*pi*x**2*rho(c,x),0,np.inf)
#print(result[0])

# checking if functions are defined properly 
result = integrate.quad(lambda x: 4*pi*x**2*(basis_satz[1]/pi)**(3./2.)*exp(-basis_satz[1]*x**2),0,np.inf)


# E_{TF}[\rho] = 4 \pi \int_0^{\infty} dr r**2 e_{TF}[\rho] 
V_TF = integrate.quad(lambda x: 4*pi*x**2*kinetische_energie(rho,c,x),0,np.inf)




## plot density and kinetic energy functional
plot = False 
if plot:
    r = [float(i/19) for i in range(-190,190)]
    values = get_values(r,c)
    EK_plot = []
    for x in r:
        EK_plot.append(kinetische_energie(rho,c,x))
    def get_values(r,c):
        values = []
        for x in r:
            values.append(rho(c,x))
        return values 
    plt.plot(r,values)
    plt.plot(r,EK_plot)
    plt.show()

spikyness = [i for i in np.linspace(4,-4,1000)] 


for spike in spikyness:
    basis_satz = [alpha for alpha in np.logspace(spike,spike-4,19)]
    V_Ke = gaussian_kernpotential_energie(c)
    V_TF = integrate.quad(lambda x: 4*pi*x**2*kinetische_energie(rho,c,x),0,np.inf)
    print(spike, spike+8, V_Ke, V_TF[0])

