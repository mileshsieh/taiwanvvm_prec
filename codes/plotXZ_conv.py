#!/home/mileshsieh/anaconda3/bin/python
import os,sys
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


def getXY(yc,xc,angle=20.0,npt=300,ny=1024):
  #sample all points
  xstart=xc-(yc*np.tan(angle/180.0*np.pi)+0.5).astype(int)
  xend=xc+((ny-yc)*np.tan(angle/180.0*np.pi)+0.5).astype(int)
  #print(xstart,xend)
  line_x=(np.linspace(xstart,xend,npt)+0.5).astype(int)
  line_y=(np.linspace(0,ny-1,npt)+0.5).astype(int)
  return line_x,line_y

if __name__ == "__main__":
    caseList=[a.split('/')[-1] for a in glob.glob('/data2/VVM/taiwanvvm_lc/taiwanvvm/lc*')]
    print(len(caseList))
    # case: 20220625
    homeDir='/data2/VVM/taiwanvvm_lc'
    case='lc_20220625'

    #get data
    dataDir=homeDir+'/%s'%case
    nt=vvm.getNT(dataDir)
    (nz,ny,nx),zc=vvm.getDimension(case,dataDir)
    tp,utp,vtp,lon,lat,lon_edge,lat_edge=vvm.getTOPO(dataDir)

    print('caseName=%s'%case)
    print('Dimensions: nx=%d, ny=%d, nz=%d, nt=%d'%(nx,ny,nz,nt))



