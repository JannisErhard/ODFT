import numpy as np
import scipy.integrate as integrate
from densities import rho
from shared_stuff import *

def kinetischer_energie_integrand(f,c,x):
    # (3 \pi^2)^{2/3} \frac{3}{10} \rho^{\frac{5}{3}} Yang-Parr p.108
    return (3*np.pi**2)**(2/3.)*3/10.*f(c,x)**(5/3.)

def kinetische_energie(f,c):
    # normal version
    res = integrate.quad(lambda x: 4*np.pi*x**2*kinetischer_energie_integrand(f,c,x),1E-1,np.inf)[0]
    res += integrate.quad(lambda x: 4*np.pi*x**2*kinetischer_energie_integrand(f,c,x),1E-5,1E-1)[0] 
    return res

def total_energy(f,c):
    norm = sum(c)
    c = [i/norm for i in c]
    E_kin = kinetische_energie(f,c) 
    E_c = gaussian_kernpotential_energie(c)
    E_tot = E_c + E_kin
    print(f"{E_c=} {E_kin=} {E_tot=}","Virial=",2*E_kin/E_c)
    return E_tot

def gaussian_kinetische_energie(c):
    # normal version
    res = integrate.quad(lambda x: 4*np.pi*x**2*kinetischer_energie_integrand(rho,c,x),1E-1,np.inf)[0]
    res += integrate.quad(lambda x: 4*np.pi*x**2*kinetischer_energie_integrand(rho,c,x),1E-5,1E-1)[0] 
    return res

def gaussian_kernpotential_energie(c):
    # use boys function trick for coulomb energy
    V_eK=0
    for c_i,alpha in zip(c,basis_satz):
        V_eK += c_i*-1*np.sqrt(4.0*alpha/np.pi)
    return V_eK

def gaussian_total_energy(c):
    norm = sum(c)
    c = [i/norm for i in c]
    E_kin = gaussian_kinetische_energie(c) 
    E_c = gaussian_kernpotential_energie(c)
    E_tot = E_c + E_kin
    print(f"{E_c=} {E_kin=} {E_tot=}","Virial=",2*E_kin/E_c)
    return E_tot
