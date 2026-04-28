import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
if __name__=='__main__':
  prec_bin=[5,10,20,30,50]
  hrList=[1,3,6]
  stepList=['+10min','+30min','+1hr']
  a=pd.read_csv('../data/caseList.94.txt',names=['caseName'])
  caseList=a.caseName.values
  csi={'north':np.load('../data/csi.north.case94.npy'),'sw':np.load('../data/csi.sw.case94.npy')}
  ncase,nconv,nt,nstep,nthd=csi['sw'].shape
  print(ncase,nconv,nt,nstep,nthd)
  #plot CSI by conv
  for area in ['sw','north']:
    plt.close()
    for k,clr in zip(range(len(hrList)),['b','orange','g']):
      for icase,case in enumerate(caseList):
        plt.plot(prec_bin,np.nanmean(csi[area][icase,k,:,-1,:].reshape((-1,5)),axis=0),lw=0.1,color=clr,alpha=0.5)
      plt.plot(prec_bin,np.nanmean(csi[area][:,k,:,-1,:].reshape((-1,5)),axis=0),color=clr,lw=2,label=f'conv_{hrList[k]}hr(+1hr)')
      #for icase,case in enumerate(caseList):
      #    plt.plot(prec_bin,np.nanmean(csi[icase,k,:,-1,:].reshape((-1,5)),axis=0),lw=0.1,color=clr,alpha=0.5)
      plt.plot(prec_bin,np.nanmean(csi[area][:,k,:,0,:].reshape((-1,5)),axis=0),color=clr,lw=1,ls=':',label=f'conv_{hrList[k]}hr(+10min)')
      plt.plot(prec_bin,np.nanmean(csi[area][:,k,:,1,:].reshape((-1,5)),axis=0),color=clr,lw=1,ls='--',label=f'conv_{hrList[k]}hr(+30min)')
    plt.legend()
    plt.xticks(prec_bin)
    plt.ylim(0,1)
    plt.ylabel('CSI')
    plt.xlabel('Precipitation Criteria (mm $hr^{-1}$)')
    plt.grid()
    plt.title(f'Critical Success Index ({area.upper()})')
    plt.savefig(f'../figures/csi.{area}.png',dpi=200)

    #plot area comparison for +1hr
    plt.close()
    for area,ls in zip(['sw','north'],['-','--']):
      for k,clr in zip(range(len(hrList)),['b','orange','g']):
        plt.plot(prec_bin,np.nanmean(csi[area][:,k,:,-1,:].reshape((-1,5)),axis=0),color=clr,lw=2,ls=ls,\
                 label=f'conv_{hrList[k]}hr({area})')
    plt.legend()
    plt.xticks(prec_bin)
    plt.ylim(0,0.4)
    plt.ylabel('CSI')
    plt.xlabel('Precipitation Criteria (mm $hr^{-1}$)')
    plt.grid()
    plt.title('Critical Success Index')
    plt.savefig('../figures/csi.all_areas.png',dpi=200)

  '''
  for k in range(3):
    plt.plot(prec_bin,np.nanmean(finalData[:,k,:,-1,:].reshape((-1,5)),axis=0),label=f'conv_{hrList[k]}hr')
  plt.legend()
  plt.xticks(prec_bin)
  plt.show()
  '''

