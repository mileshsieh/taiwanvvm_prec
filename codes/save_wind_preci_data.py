import os,sys,glob
import time
import numpy as np
import multiprocessing
import xarray as xr
import vvmDataUtil as vvm
from matplotlib import pyplot as plt

def get_SFC_Field(tt,case,dataDir,topo,utopo,vtopo):
  #print("\tCalculating timestep %03d"%tt)
  fname=f'{dataDir}/archive/{case}.C.Surface-{tt:06}.nc'
  sprec=xr.open_dataset(fname).sprec.values[0]*3600 # unit: mm/hr
  usfc,vsfc=vvm.getSfcWind_intp(dataDir,case,topo,utopo,vtopo,tt)
  return tt,sprec,usfc,vsfc

if __name__ == "__main__":
  nProc=32
  case='tpe_20100802_at001'
  dataDir='/data2/VVM/taiwanvvm_summer/'+case
  print('caseName=%s'%case)

  nt=vvm.getNT(dataDir)
  (nz,ny,nx),zc=vvm.getDimension(case,dataDir)
  topo,utopo,vtopo,lon,lat,lon_edge,lat_edge=vvm.getTOPO(dataDir)
  np.save('../data/zc.npy',zc.filled(0.0))

  pool=multiprocessing.Pool(nProc)
  #for all the runs
  caseList=[a.split('/')[-1] for a in glob.glob('/data2/VVM/taiwanvvm_summer/*_20*')]
  #caseList=[a.split('/')[-1] for a in glob.glob('/data3/VVM/taiwanvvm_summer_ats/*_20*')]
  #caseList=[a.split('/')[-1] for a in glob.glob('/data2/VVM/taiwanvvm_lc/*_20*')]
  #caseList=['tpe_20080715_at006']
  #sfcList=[a.split('/')[-1].split('.')[0] for a in glob.glob('../data/*.sfc.npy')]
  #sfcList=[]
  for c in caseList:
    if c in sfcList:
      print(f'skip {c}')
      continue
    else:
      dataDir='/data2/VVM/taiwanvvm_summer/'+c
      #dataDir='/data3/VVM/taiwanvvm_summer_ats/'+c
      #dataDir='/data2/VVM/taiwanvvm_lc/'+c
      nt=vvm.getNT(dataDir)
      print(c,nt)
      if nt<145:
        continue

      tStart=time.time()
      print('Calculating %d timesteps using %d processes...'%(nt,nProc))

      #put the arguments into tasks list
      tasks=[]
      #for tt in tmList:
      for tt in range(nt):
        tasks.append((tt,c,dataDir,topo,utopo,vtopo))
      results=[pool.apply_async(get_SFC_Field,t) for t in tasks]

      final=np.zeros((nt,3,ny,nx))
      for result in results:
        rt,sprec,usfc,vsfc=result.get()
        final[rt]=np.stack([sprec,usfc,vsfc])
        #print("Result: got timestep %03d "%rt)
      np.save(f'../data/{c}.sfc.npy',final)

      tEnd=time.time()
      print('Calculating %d timesteps using %d processes...It costed %f sec'%(nt,nProc,tEnd-tStart))

 
