import os,sys,glob
import time
import numpy as np
import multiprocessing
import xarray as xr
from matplotlib import pyplot as plt

def getPrecp(tt,case,dataDir):
  print("\tCalculating timestep %03d"%tt)
  fname=f'{dataDir}/archive/{case}.C.Surface-{tt:06}.nc'
  sprec=xr.open_dataset(fname).sprec.values[0]*3600 # unit: mm/hr
  return tt,sprec[sprec>=0.1].mean()

if __name__ == "__main__":
  nProc=16
  case='tpe_20100802_at001'
  dataDir='/data2/VVM/taiwanvvm_summer/'+case
  nt=145
  print('caseName=%s'%case)

  tStart=time.time()
  pool=multiprocessing.Pool(nProc)
  print('Calculating %d timesteps using %d processes...'%(nt,nProc))

  #put the arguments into tasks list
  tasks=[]
  #for tt in tmList:
  for tt in range(nt):
    tasks.append((tt,case,dataDir))
  results=[pool.apply_async(getPrecp,t) for t in tasks]

  meanPrec=np.zeros(nt)
  for result in results:
    rt,sprec=result.get()
    meanPrec[rt]=sprec
    print("Result: got timestep %03d "%rt)

  plt.close()
  plt.plot(np.arange(nt)/6.0,meanPrec)
  plt.xlim(0,24)
  plt.xlabel('Local Hour')
  plt.ylabel('Mean Precipitation (mm/hr)')
  plt.grid()

  tEnd=time.time()
  print('Calculating %d timesteps using %d processes...It costed %f sec'%(nt,nProc,tEnd-tStart))

 
