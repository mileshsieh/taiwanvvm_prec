import os,sys,glob
import time
import numpy as np
import multiprocessing
import xarray as xr
import vvmDataUtil as vvm
from matplotlib import pyplot as plt
from scipy.signal import gaussian
from scipy.ndimage import convolve1d

def conv_SFC_Field(npyfile,hr):
  c=npyfile.split('/')[-1].split('.')[0]
  npts=hr*6+1
  wgt=gaussian(npts,hr)
  wgt=wgt/wgt.sum()

  sfcData=np.load(npyfile)
  #final=convolve1d(sfcData,wgt,axis=0,mode='constant',cval=0.0)
  final=convolve1d(sfcData,wgt,axis=0,mode='wrap')
  np.save(f'../data/ae_data/{c}.conv.{hr}hr.npy',final)
  return c,hr

def saveCaseList():
    caseList=sorted([a.split('/')[-1].split('.')[0] for a in glob.glob('../data/ae_data/*.conv.1hr.npy')])
    with open('../data/caseList.txt','w') as outF:
      for c in caseList:
        print(c,file=outF)

if __name__ == "__main__":
  nProc=32
  pool=multiprocessing.Pool(nProc)
  #for all the runs
  dataList=glob.glob('../data/*sfc*npy')
  #caseList=['tpe_20080715_at006']

  #skip calcualted cases
  #convCaseList=[a.split('/')[-1].split('.')[0] for a in glob.glob('../data/ae_data/*conv.9hr.npy')]
  convCaseList=[]

  tasks=[]
  ncase=0
  for f in dataList:
    if f.split('/')[-1].split('.')[0] in convCaseList:
      print('skip case:',f.split('/')[-1].split('.')[0])
      continue
    else:
      print('convolve case:',f.split('/')[-1].split('.')[0])
      ncase=ncase+1
      #for hr in [1,2,3,6]:
      for hr in [1,2,3,4,5,6,9,12,18,24]:
        tasks.append((f,hr))

  tStart=time.time()
  print('Calculating using %d processes...'%nProc)
  results=[pool.apply_async(conv_SFC_Field,t) for t in tasks]
  for result in results:
    rc,rhr=result.get()
    #print(f"Result: got {rc} with {rhr} hour convolued")
  tEnd=time.time()
  print('Calculating %d cases using %d processes...It costed %f sec'%(ncase,nProc,tEnd-tStart))
  saveCaseList()
 
