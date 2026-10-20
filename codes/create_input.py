import numpy as np
import pandas as pd
import glob,os
import multiprocessing

xyLL={
        'sw':[262,320],
        'north':[481,673],
        'south':[323,120],
        }
npts=256

def getCaseList(ncase):
  if ncase==0:
    a=pd.read_csv('../data/caseList.txt',names=['caseName'])
  else:
    a=pd.read_csv(f'../data/caseList.{ncase}.txt',names=['caseName'])
  caseList=a.caseName.values
  return caseList

def _getData(i,c,dataDir,hr,area,ts,te,xyLL,npts):
  if hr>0:
    data=np.load(f'{dataDir}{c}.conv.{hr}hr.npy')[ts:te,:,\
          xyLL[area][1]:xyLL[area][1]+npts,xyLL[area][0]:xyLL[area][0]+npts]
  else:
    data=np.load(f'{dataDir}{c}.sfc.npy')[ts:te,:,\
          xyLL[area][1]:xyLL[area][1]+npts,xyLL[area][0]:xyLL[area][0]+npts]

  wind_max=data[:,1:,:,:].max()
  prec_max=data[:,0,:,:].max()
  print(f'getting data {i}:{c}  prec={prec_max}, wind={wind_max}',data.shape)
  return i,data.astype(np.float32)

def load_convolution_data(tstart,tend,hr,area,ncase,dataDir):
  nProc=12
  #setup data directory
  #load data
  caseList=getCaseList(ncase)
  pool=multiprocessing.Pool(nProc)
  tasks=[]
  for i in range(len(caseList)):
    tasks.append((i,caseList[i],dataDir,hr,area,tstart,tend,xyLL,npts))
  results=[pool.apply_async(_getData,t) for t in tasks]

  final=np.zeros((te-ts+1,len(caseList),3,npts,npts))
  for result in results:
      (ri,data)=result.get()
      wind_max=data[:,1:,:,:].max()
      prec_max=data[:,0,:,:].max()
      print(f'saving data {ri}  prec={prec_max}, wind={wind_max}',data.shape)
      final[:,ri,:,:,:]=data

  print(final.shape)
  return caseList,final


ts=48
te=120
if __name__=='__main__':
  ncase=50
  #for hr in range(1,7):
  for hr in [24,]:
    for area in xyLL.keys():
      if hr==0:
          dataDir='../data/'
      else:
          dataDir='../data/ae_data/'
      caseList,final=load_convolution_data(ts,te+1,hr,area,ncase,dataDir)
      np.save('../data/input/input.t%dto%d.ncase%d.%s.conv_%dhr.npy'%(ts,te,ncase,area,hr),final)

