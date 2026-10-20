import os,sys,glob
import time
import pandas as pd
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
  case='sw_VVRF001_20250728T0000_20250804T0000'
  dataDir='/data2/VVM/taiwanvvm_summer_vvrf/'+case
  print('caseName=%s'%case)

  nt=vvm.getNT(dataDir)
  (nz,ny,nx),zc=vvm.getDimension(case,dataDir)
  topo,utopo,vtopo,lon,lat,lon_edge,lat_edge=vvm.getTOPO(dataDir)
  topo=topo[:1024,:1024]
  utopo=utopo[:1024,:1024]
  vtopo=vtopo[:1024,:1024]
  lon=lon[:1024]
  lat=lat[:1024]
  #np.save('../data/zc.vvrf.npy',zc.filled(0.0))
  #np.save('../data/lon.vvrf.npy',lon.filled(0.0))
  #np.save('../data/lat.vvrf.npy',lat.filled(0.0))
  #np.save('../data/topo.vvrf.npy',topo.filled(0.0))

  pool=multiprocessing.Pool(nProc)
  #for vvrf runs
  for c in ['sw_VVRF001_20250728T0000_20250804T0000',]:
      dataDir='/data2/VVM/taiwanvvm_summer_vvrf/'+c
      nt=vvm.getNT(dataDir)
      print(c,nt)
      stime=pd.to_datetime('20250728')

      nday=8
      nt=145
      for iday in range(nday):
        tStart=time.time()
        print('Calculating %d timesteps using %d processes...'%(nt,nProc))
        ymd=(stime+pd.Timedelta(iday,'D')).strftime('%Y%m%d')
        #put the arguments into tasks list
        tasks=[]
        #for tt in tmList:
        for tt in range(iday*144,iday*144+145):
          tasks.append((tt,c,dataDir,topo,utopo,vtopo))
        results=[pool.apply_async(get_SFC_Field,t) for t in tasks]

        final=np.zeros((nt,3,ny,nx))
        for result in results:
          rt,sprec,usfc,vsfc=result.get()
          final[rt-(iday*144)]=np.stack([sprec,usfc,vsfc])
          #print("Result: got timestep %03d "%rt)
        np.save(f'../data/sw_{ymd}_VVRF001.sfc.npy',final)

        tEnd=time.time()
        print('Calculating %d timesteps using %d processes...It costed %f sec'%(nt,nProc,tEnd-tStart))

 
