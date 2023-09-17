import numpy as np

basis_set_size = 10
n = basis_set_size
basis_satz = [1.7642452031482*2**i for i in np.linspace(-4,5,n)]
basis_satz = [1.7642452031482/4.,1.7642452031482/2.,1.7642452031482,1.7642452031482*2,1.7642452031482*4]
