import numpy as np

tke=np.load('/data/mileshsieh/turbulence_map/data/processed/tke/lc_20220625.tke.npy')

for tt in range(tke.shape[0]):
    print(tt)
    np.save('/data/mileshsieh/turbulence_map/data/processed/tke/lc_20220625.tke,t%03d.npy'%tt,tke[tt])

