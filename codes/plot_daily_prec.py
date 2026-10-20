import numpy as np
from matplotlib import pyplot as plt
import pickle,glob
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

ws_levels = [0.0, 1.5, 3.3, 5.4, 7.9, 10.7, 13.8, 17.1, 25.0]


# 依照附圖近似配置的 8 個顏色
ws_colors = [
    "#9BCBF3",  # <=1
    "#C7ECF4",  # 2
    "#C8E2B9",  # 3
    "#A9D97D",  # 4
    "#FFAA00",  # 5
    "#FF4E0B",  # 6
    "#EC3294",  # 7
    "#EDB7EE",  # >=8
]

pltCfg={
        #'PBLH':['PBL Height Above Ground','PBL Height(m)',np.linspace(0.0,2500.0,101,endpoint=True),'max'],
        'sprec':['Precipitation','Precipitation(mm/hr)',rain_levels,rain_colors,'max'],
        'maxws':['Max WS','Wind Speed (m/s)',ws_levels,ws_colors,'max'],
        }


def plotTW(var,ax,lon,lat,topo,varData,title,rtitle,ltitle):
    lvls=pltCfg[var][2]
    clrs=pltCfg[var][3]
    ext=pltCfg[var][4]
    #ny,nx=topo.shape
    #xx,yy=np.meshgrid(lon,lat)
    #ws=np.sqrt(varData[1,:,:]**2+varData[2,:,:]**2)
    #cstopo=ax.contourf(lon,lat,topo,levels=np.linspace(0,1500,11),cmap='Greys',extend='max')
    cs=ax.contourf(lon,lat,varData,levels=lvls,colors=clrs,extend=ext)
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
  caseList=[a.split('/')[-1].split('.')[0] for a in sorted(glob.glob('../data/tpe_*_LCD*.npy'))]
  area='tw'
  topo=np.load('../data/topo.npy')
  lon=np.load('../data/lon.npy')
  lat=np.load('../data/lat.npy')
  zc=np.load('../data/zc.npy')
  topo=zc[topo.astype(np.int32)]
  
  for c in caseList:
      sim=np.load(f'../data/{c}.sfc.npy')
      prec=sim[:,0,:,:]
      #u=sim[:,1,:,:]
      #v=sim[:,2,:,:]
      #ws=np.sqrt(u*u+v*v)
      #max_ws=np.max(ws,axis=0)

      nt,ny,nx=prec.shape
      print(c,nt,ny,nx)
      prec=np.sum(prec,axis=0)/nt*24
      prec=np.where(topo>0,prec,np.nan)
      #max_ws=np.where(topo>0,max_ws,np.nan)
      varData={'sprec':prec} #,'maxws':max_ws}

      var='sprec'
      plt.close()
      fig=plt.figure(figsize=(8,10))
      ax=plt.subplot()
      cs=plotTW(var,ax,lon[xyLL[area][0]:xyLL[area][1]],lat[xyLL[area][2]:xyLL[area][3]],\
                   topo[xyLL[area][2]:xyLL[area][3],xyLL[area][0]:xyLL[area][1]],\
                   varData[var][xyLL[area][2]:xyLL[area][3],xyLL[area][0]:xyLL[area][1]],\
                   #prec[xyLL[area][2]:xyLL[area][3],xyLL[area][0]:xyLL[area][1]],\
                   '',c.split('_')[1],c.split('_')[0])

      #plt.tight_layout()
      if var=='maxws':
          cbar=plt.colorbar(cs)
          # 風級標籤放在每個色塊中央
          level_centers=[(ws_levels[i] + ws_levels[i + 1]) / 2 for i in range(len(ws_levels) - 1)]
          level_labels = ["≤ 1", "2", "3", "4", "5", "6", "7", "≥ 8"]

          cbar.set_ticks(level_centers)
          cbar.set_ticklabels(level_labels)
      else:
          cbar=plt.colorbar(cs,ticks=pltCfg[var][2])
      cbar.set_label(pltCfg[var][1],fontsize=15)
      plt.savefig(f'../figures/daily_{var}.{c}.png',dpi=300)
      '''
      var='maxws'
      plt.close()
      fig=plt.figure(figsize=(8,10))
      ax=plt.subplot()
      cs=plotTW(var,ax,lon[xyLL[area][0]:xyLL[area][1]],lat[xyLL[area][2]:xyLL[area][3]],\
                   topo[xyLL[area][2]:xyLL[area][3],xyLL[area][0]:xyLL[area][1]],\
                   varData[var][xyLL[area][2]:xyLL[area][3],xyLL[area][0]:xyLL[area][1]],\
                   #prec[xyLL[area][2]:xyLL[area][3],xyLL[area][0]:xyLL[area][1]],\
                   '',c.split('_')[1],c.split('_')[0])

      #plt.tight_layout()
      if var=='maxws':
          cbar = fig.colorbar(cs,orientation="vertical",boundaries=ws_levels,spacing="proportional",extend="max",pad=0.05,)

          # Wind-speed ticks are placed at color boundaries
          ws_ticks = ws_levels[1:-1]
          cbar.set_ticks(ws_ticks)
          cbar.set_ticklabels(["1.5","3.3","5.4","7.9","10.7","13.8","17.1",])

          cbar.ax.yaxis.set_ticks_position("right")
          cbar.ax.yaxis.set_label_position("right")
          cbar.ax.set_ylabel(r"Wind speed (m s$^{-1}$)", fontsize=13, labelpad=12)

      else:
          cbar=plt.colorbar(cs,ticks=pltCfg[var][2])
          cbar.set_label(pltCfg[var][1],fontsize=15)
      plt.savefig(f'../figures/daily_{var}.{c}.png',dpi=300)
      '''
