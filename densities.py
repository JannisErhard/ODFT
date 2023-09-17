import numpy as np
from shared_stuff import *

def gaussian(alpha,x):
    # keep density normalized
    return (alpha/np.pi)**(3/2.)*np.exp(-alpha*x**2)

def rho(c,x):
    # density as expansion in basis set 
    p = 0
    for c_i,alpha in zip(c,basis_satz):
        p += c_i*gaussian(alpha,x)
    return p
