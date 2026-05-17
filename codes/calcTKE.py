#!/home/mileshsieh/anaconda3/bin/python
import os,sys,glob
import time
import numpy as np
import pandas as pd
import vvmDataUtil_v2021 as vvm
import multiprocessing
import matplotlib
from matplotlib import pyplot as plt
Cp=1004.0
g=9.8
Lv=2264760.0
R=8.314

def calcAll(tt,case,dataDir,zc,topo):
  print("\tCalculating timestep %03d"%tt)
  u=vvm.getData(dataDir,'u','exp',tt)
  v=vvm.getData(dataDir,'v','exp',tt)
  w=vvm.getData(dataDir,'w','exp',tt)

  for d in [u,v,w]:
    for k in range(topo.max().astype(np.int)):
      d[k,:,:]=np.where(topo.astype(np.int)>k,np.nan,d[k,:,:])
      
  vke=w*w
  tke=calcTKE(u,v,w)
  #calculate integral top
  nz,ny,nx=tke.shape
  tke2D=tke.reshape((nz,ny*nx))
  # max decrease of TKE: too sharp
  #del_tke2D=tke2D[1:,:]-tke2D[:-1,:]
  idx_max_tke=np.nanargmax(tke2D,axis=0)
  # decreased to p50
  int_top=(idx_max_tke+np.array([np.where(tke2D[idx_max_tke[k]:,k]<=np.percentile(tke2D[idx_max_tke[k]:,k],50))[0][0] for k in range(tke2D.shape[1])])).reshape((ny,nx))
  int_tke=vertical_integral(tke,zc,int_top)
  int_vke=vertical_integral(vke,zc,int_top)
  
  return tt,int_tke,int_vke,idx_max_tke.reshape((ny,nx))

def vertical_integral(data3D,zc,int_top):
  rho=vvm.getFort98Info(dataDir,'exp','RHO')
  ztop=zc[-1]
  zc_extended=np.append(zc,zc[-1]+zc[-1]-zc[-2])
  dz=(zc_extended[1:]-zc_extended[:-1]).reshape((-1,1,1))
  #print(ztop,idx_ztop,zc_extended.shape,dz.shape,rho.shape)
  for k in range(zc.shape[0]):
    data3D[k,:,:]=np.where(int_top>k,data3D[k,:,:],np.nan)
  
  v_int_data=np.nansum(data3D*dz*rho.reshape((-1,1,1)),axis=0)*1e-3
  #v_int_data=np.nansum(data3D*dz*rho.reshape((-1,1)),axis=0)
  # unit: kJ/m2
  return v_int_data

def calcTKE(u,v,w):
  mU=np.nanmean(np.nanmean(u,axis=-1),axis=-1).reshape((-1,1,1))
  mV=np.nanmean(np.nanmean(v,axis=-1),axis=-1).reshape((-1,1,1))
  tke=0.5*((u-mU)*(u-mU)+(v-mV)*(v-mV)+w*w)
  return tke

def getCrossLine(topo):
  #plot yz cross section over highest topo
  yy,xx=divmod(np.argmax(topo),1024)
  lx,ly=getXY(yy,xx)
  return lx,ly

def getXY(yc,xc,angle=20.0,npt=300,ny=1024):
  #sample all points
  xstart=xc-(yc*np.tan(angle/180.0*np.pi)+0.5).astype(int)
  xend=xc+((ny-yc)*np.tan(angle/180.0*np.pi)+0.5).astype(int)
  #print(xstart,xend)
  line_x=(np.linspace(xstart,xend,npt)+0.5).astype(int)
  line_y=(np.linspace(0,ny-1,npt)+0.5).astype(int)
  return line_x,line_y

if __name__ == "__main__":
  tStart=time.time()
  caseList=[a.split('/')[-1] for a in glob.glob('/data3/mog/taiwanvvm/tpe*dty')]
  print(len(caseList))
  #case=caseList[0]
  case='tpe20110816dtyP3K'
  dataDir=vvm.vvmDir+'/%s'%case
  nt=vvm.getNT(dataDir)
  (nz,ny,nx),zc=vvm.getDimension('exp',dataDir)
  topo,lon,lat=vvm.getTOPO(dataDir)

  print('caseName=%s'%case)
  print('Dimensions: nx=%d, ny=%d, nz=%d, nt=%d'%(nx,ny,nz,nt))
  #rt,tke,vke,maxH=calcAll(0,case,dataDir,zc,topo)
  
  nProc=16
  #nt=32
  pool=multiprocessing.Pool(nProc)
  #print('Calculating %03d timestep using %d processes...'%(len(tmList),multiprocessing.cpu_count()))
  print('Calculating %d timesteps using %d processes...'%(nt,nProc))

  #put the arguments into tasks list
  tasks=[]
  for tt in range(nt):
    tasks.append((tt,case,dataDir,zc,topo))

  results=[pool.apply_async(calcAll,t) for t in tasks]
  
  tke_all=np.zeros((nt,ny,nx))
  vke_all=np.zeros((nt,ny,nx))
  mHgt_all=np.zeros((nt,ny,nx))
  for result in results:
    rt,tke,vke,maxH=result.get()
    tke_all[rt,:,:]=tke
    vke_all[rt,:,:]=vke
    mHgt_all[rt,:,:]=maxH
    print('Result: timestep %03d is collected'%rt)

  np.save('./data/%s.tke.npy'%case,tke_all)
  np.save('./data/%s.vke.npy'%case,vke_all)
  np.save('./data/%s.maxH.npy'%case,mHgt_all)
  
  tEnd=time.time()
  mm,ss=divmod(int(tEnd-tStart),60)
  hh,mm=divmod(mm,60)
  print('Calculating %s %d timesteps using %d processes...It costed %d:%02d:%02d'%(case,nt,nProc,hh,mm,ss)) 
  
