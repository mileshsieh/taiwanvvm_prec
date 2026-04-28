import numpy as np
from matplotlib import pyplot as plt
import pickle
import vvmDataUtil as vvm
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.colors as mc
import matplotlib
from mpl_toolkits.axes_grid1 import AxesGrid
matplotlib.rc('xtick',labelsize=10)
matplotlib.rc('ytick',labelsize=10)

with open('PrecipitationColorTable.pickle', 'rb') as fc:
  rain_levels, rain_colors = pickle.load(fc)

top=rain_levels.pop()
pltCfg={'pAbd':['Pseudo Albedo[case=%s Local Time=%s]','Pseudo Albedo',np.linspace(0.0,1.0,51,endpoint=True),'max'],
        'CBH':['Cloud Base Height Above Ground','Cloud Base Height(m)',np.linspace(0.0,7500.0,101,endpoint=True),'max'],
        'CTH':['Cloud Top Height Above Ground','Cloud Top Height(m)',np.linspace(0.0,7500.0,101,endpoint=True),'max'],
        'PBLH':['PBL Height Above Ground','PBL Height(m)',np.linspace(0.0,2500.0,101,endpoint=True),'max'],
        'tr01':['Tracer01 Con. on the Ground','Tracer Con.($g/m^3$)',np.linspace(0.0,0.01,101,endpoint=True),'max'],
        'sprec':['Precipitation','Precipitation(mm/hr)',rain_levels,'max'],
        'CAPE':['CAPE','CAPE(J/kg)',np.linspace(0.0,3500.0,71,endpoint=True),'max'],
        }

def plotPrec(ax,lon,lat,topo,varData,title,rtitle,ltitle):
    #ny,nx=topo.shape
    #xx,yy=np.meshgrid(lon,lat)
    #ws=np.sqrt(varData[1,:,:]**2+varData[2,:,:]**2)
    #cstopo=ax.contourf(lon,lat,topo,levels=np.linspace(0,1500,11),cmap='Greys',extend='max')
    cs=ax.contourf(lon,lat,varData,levels=rain_levels,colors=rain_colors,extend='max')
    ax.contour(lon,lat,topo,levels=[0.5,],colors=['k',],linewidths=[2,],zorder=6)
    ax.contour(lon,lat,topo,levels=[200,],colors=['k',],linewidths=[0.7,],zorder=6)
    ax.set_xlim(lon[0],lon[-1])
    ax.set_ylim(lat[0],lat[-1])
    #ax.xaxis.set_major_locator(ticker.NullLocator())
    #ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.set_title(rtitle,fontsize=10,loc='right')
    ax.set_title(ltitle,fontsize=10,loc='left')
    ax.set_title(title,fontsize=18,loc='center')
    ax.grid()
    return cs    

def plotBoxes(zc,topo,xyLL,npts):
    plt.close()
    cstopo=plt.contourf(np.where(topo>0,topo,np.nan),levels=np.linspace(0,1500,11),cmap='Greens',extend='max')
    plt.contour(topo,levels=[0.5,],colors=['k',],zorder=6)
    #for a in xyLL.keys():
    for a in ['sw']:
        xx=xyLL[a][0]
        yy=xyLL[a][1]
        plt.plot([xx,xx+npts,xx+npts,xx,xx],[yy,yy,yy+npts,yy+npts,yy],'r-')
        plt.annotate(a,(xx,yy),(xx+20,yy+npts-50),color='r')

    plt.gca().xaxis.set_major_locator(ticker.NullLocator())
    plt.gca().yaxis.set_major_locator(ticker.NullLocator())
    plt.savefig('../figures/areas.png')
    return

xyLL={
        'sw':[262,320],
        'north':[481,673],
        'south':[323,120],
        'tw':[262,737,120,930],
        }

if __name__ == "__main__":
  case='sw_20190815_sw005'
  caseList=['sw_20190815_sw005','sw_20190815_sw005_n6uvtq',\
          'sw_20190815_sw005_nudging6hr','tpe_20100912_at002']
  area='tw'
  topo=np.load('../data/topo.npy')
  lon=np.load('../data/lon.npy')
  lat=np.load('../data/lat.npy')
  zc=np.load('../data/zc.npy')
  topo=zc[topo.astype(np.int32)]
  '''
  for c in caseList:
      prec=np.load(f'../data/{c}.sfc.npy')[:,0,:,:]
      nt,ny,nx=prec.shape
      print(c,nt,ny,nx)
      prec=np.sum(prec,axis=0)/nt*24
      prec=np.where(topo>0,prec,0.0)
      plt.close()
      fig=plt.figure(figsize=(8,10))
      ax=plt.subplot()
      cs=plotPrec(ax,lon[xyLL[area][0]:xyLL[area][1]],lat[xyLL[area][2]:xyLL[area][3]],\
                     topo[xyLL[area][2]:xyLL[area][3],xyLL[area][0]:xyLL[area][1]],\
                     prec[xyLL[area][2]:xyLL[area][3],xyLL[area][0]:xyLL[area][1]],\
                     '',c.split('_')[1],c.split('_')[0])

      #plt.tight_layout()
      cbar=plt.colorbar(cs,ticks=pltCfg['sprec'][2])
      cbar.set_label(pltCfg['sprec'][1],fontsize=15)
      plt.savefig('../figures/daily_prec.%s.png'%c,dpi=300)

  '''
