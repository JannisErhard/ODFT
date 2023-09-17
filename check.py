from math import exp, pi
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt  
import scipy.integrate as integrate

from shared_stuff import basis_satz, n
from energies import kinetische_energie, gaussian_kernpotential_energie, gaussian_total_energy, total_energy
from densities import rho, gaussian

# if V(r) = a r^n then 2 <T> = n <V>

def H_den(x):
    return (exp(-abs(x)))**2/pi

def difference_integral(c):
    norm = sum(c)
    c = [i/norm for i in c]
    res = integrate.quad(lambda x: 4*pi*x**2*abs(rho(c,x)-H_den(x)),1E-1,np.inf)[0]
    res += integrate.quad(lambda x: 4*pi*x**2*abs(rho(c,x)-H_den(x)),1E-5,1E-1)[0] 
    return res



c=[float(i) for i in range(1,n+1)]
norm = sum(c)
c_0 = [i/norm for i in c]

l_Optimize = True
l_fit_densities = True
plot = True
spike_test=False
l_1D_test=False

# checking if rho is defined properly 
result = integrate.quad(lambda x: 4*pi*x**2*rho(c,x),0,np.inf)

# checking if functions are defined properly 
result = integrate.quad(lambda x: 4*pi*x**2*(basis_satz[1]/pi)**(3./2.)*exp(-basis_satz[1]*x**2),0,np.inf)

# E_{TF}[\rho] = 4 \pi \int_0^{\infty} dr r**2 e_{TF}[\rho] 
V_TF =  kinetische_energie(rho,c)



# run through a number of basis sets, desribing very dense to very disperse charge blobs for Hydrogen, to plot sharpness against energy, to visualize the minimum 
if spike_test:
    spikyness = [i for i in np.linspace(1,-4,1000)] 
    for spike in spikyness:
        basis_satz = [alpha for alpha in np.logspace(spike+1.5,spike-1.5,19)]
        V_Ke = gaussian_kernpotential_energie(c)
        V_TF = kinetische_energie(rho,c)
        print(spike, spike+8, V_Ke, V_TF, )

if l_1D_test:
    spikyness = [i for i in np.logspace(-4,10,1000)] 
    #  just one function, steepness test 
    for spike in spikyness:
        basis_satz = [spike]
        c = [1.0]
        V_Ke = gaussian_kernpotential_energie(c)
        V_TF = kinetische_energie(rho,c)
        print(spike,V_Ke, V_TF, gaussian_total_energy(c))


# optimise the basis set coeficients to find the energy minimum for Hydrogen

if l_fit_densities:
    bnds = ((0, 1.0),)*n # only use positive numbers since there can not be any negative region in the density
    c0 = np.array(c)
    res = minimize(difference_integral,c,bounds=bnds)

    norm_final = sum(res.x)
    c_f = [i/norm_final for i in res.x]
    print("SUM CF:",sum(c_f))
    if True:
        for i,j in zip(c_f,basis_satz):
            print(i,j)
    print("final density integral:",integrate.quad(lambda x: 4*pi*x**2*rho(c_f,x),0,np.inf))


if l_Optimize:
    c0 = np.array(c)
    print(total_energy(rho,c), total_energy(rho,c0))
    bnds = ((0, 1.0),)*n # only use positive numbers since there can not be any negative region in the density
    res = minimize(gaussian_total_energy,c,bounds=bnds)
    print(gaussian_total_energy(res.x))
    
    norm_final = sum(res.x)
    c_f = [i/norm_final for i in res.x]

    print("SUM CF:",sum(c_f))

    if True:
        for i,j in zip(c_f,basis_satz):
            print(i,j)

    print(integrate.quad(lambda x: 4*pi*x**2*H_den(x),0,np.inf))
    print("final density integral:",integrate.quad(lambda x: 4*pi*x**2*rho(c_f,x),0,np.inf))

if plot:
    print("E_kin, Hden",integrate.quad(lambda x: 4*pi*x**2*(3*pi**2)**(2/3.)*3/10.*H_den(x)**(5/3.) ,0,np.inf))
    r = [float(i/19) for i in range(-190,190)]
    rho_initial = [rho(c_0,x) for x in r ]
    rho_final = [rho(c_f,x) for x in r ]
    Reference = [H_den(x) for x in r ]
    Gauss = [gaussian(1,x) for x in r ]
    #plt.plot(r,rho_initial)
    plt.plot(r,rho_final)
    plt.plot(r,Reference)
    plt.plot(r,Gauss)
    plt.show()
