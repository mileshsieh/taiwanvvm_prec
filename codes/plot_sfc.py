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
matplotlib.rc('xtick',labelsize=8)
matplotlib.rc('ytick',labelsize=8)

with open('PrecipitationColorTable.pickle', 'rb') as fc:
  rain_levels, rain_colors = pickle.load(fc)

pltCfg={'pAbd':['Pseudo Albedo[case=%s Local Time=%s]','Pseudo Albedo',np.linspace(0.0,1.0,51,endpoint=True),'max'],
        'CBH':['Cloud Base Height Above Ground','Cloud Base Height(m)',np.linspace(0.0,7500.0,101,endpoint=True),'max'],
        'CTH':['Cloud Top Height Above Ground','Cloud Top Height(m)',np.linspace(0.0,7500.0,101,endpoint=True),'max'],
        'PBLH':['PBL Height Above Ground','PBL Height(m)',np.linspace(0.0,2500.0,101,endpoint=True),'max'],
        'tr01':['Tracer01 Con. on the Ground','Tracer Con.($g/m^3$)',np.linspace(0.0,0.01,101,endpoint=True),'max'],
        'sprec':['Precipitation','Precipitation(mm/hr)',rain_levels,'max'],
        'CAPE':['CAPE','CAPE(J/kg)',np.linspace(0.0,3500.0,71,endpoint=True),'max'],
        }

def plotSfc(ax,lon,lat,zc,topo,varData,step,title,rtitle,ltitle):
    #ny,nx=topo.shape
    #xx,yy=np.meshgrid(lon,lat)
    ws_plot=np.where(np.sqrt(varData[1,:,:]**2+varData[2,:,:]**2)>=8.0,1.0,0.0)
    cstopo=ax.contourf(lon,lat,zc[topo.astype(np.int32)],levels=np.linspace(0,1500,11),cmap='Greys',extend='max')
    cs=ax.contourf(lon,lat,varData[0],levels=rain_levels,colors=rain_colors,extend='max',alpha=0.6)
    q1=ax.quiver(lon[::step],lat[::step],varData[1,::step,::step],varData[2,::step,::step],ws_plot[::step,::step],\
            scale=120,scale_units='width',angles='xy',zorder=4,width=0.004,\
            cmap=mc.ListedColormap(['black', 'red']))
    #q1=ax.quiver(lon[::step],lat[::step],varData[1,::step,::step],varData[2,::step,::step],\
    #        scale=120,scale_units='width',angles='xy',zorder=4,color='k',width=0.004,)
    ax.contour(lon,lat,topo,levels=[0.5,],colors=['k',],zorder=6)
    ax.set_xlim(lon[0],lon[-1])
    ax.set_ylim(lat[0],lat[-1])
    #ax.xaxis.set_major_locator(ticker.NullLocator())
    #ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.set_title(rtitle,fontsize=10,loc='right')
    ax.set_title(ltitle,fontsize=10,loc='left')
    ax.set_title(title,fontsize=18,loc='center')
    #ax.grid()
    ax.set_aspect('equal')
    return cs,cstopo,q1    

xyLL={
        'SW':[262,320],
        'north':[481,673],
        'south':[323,120],
        'NW':[400,641],
        }

if __name__ == "__main__":
  #case='tpe_20100802_at001'
  case='tpe_20100717_at017'
  case='tpe_20180828_sw004'
  case='sw_20150727_at024'
  case='sw_20150727_at024'
  case='lc_20220625'
  dataDir=vvm.vvmDir+'/%s'%case
  '''
  nt=vvm.getNT(dataDir)
  (nz,ny,nx),zc=vvm.getDimension(case,dataDir)
  topo,utopo,vtopo,lon,lat,lon_edge,lat_edge=vvm.getTOPO(dataDir)
  print('caseName=%s'%case)
  print('Dimensions: nx=%d, ny=%d, nz=%d'%(nx,ny,nz))
  '''
  topo=np.load('../data/topo.npy')
  lon=np.load('../data/lon.npy')
  lat=np.load('../data/lat.npy')
  zc=np.load('../data/zc.npy')

  print('topo:',topo.min(),topo.max())
  print('lat:',lat.min(),lat.max())
  print('lon:',lon.min(),lon.max())
  npts=256

  dt=10 #min
  ymd=case.split('_')[1]
  stime=np.datetime64('%s-%s-%s 00:00:00'%(ymd[:4],ymd[4:6],ymd[6:]))

  varData=np.load(f'../data/{case}.sfc.npy')
  #areaList=['north','sw','south']
  areaList=['NW','SW',]
  tstart=48
  tend=120
  step=10
  #for tt in [81]:
  for tt in range(tstart,tend+1):
    #for area in xyLL.keys():
    a=stime+np.timedelta64(dt*tt,'m')
    hhmm=pd.to_datetime(str(a)).strftime('%H:%M')
    print(hhmm)

    plt.close()
    fig=plt.figure(figsize=(len(areaList)*3+1,4))
    #grid=AxesGrid(fig,111,nrows_ncols=(1,3),axes_pad=(0.1,0.4),share_all=False)

    for i,area in enumerate(areaList):
      print(i,area)
      ax=plt.subplot(1,len(areaList),i+1)
      cs,cstopo,q1=plotSfc(ax,lon[xyLL[area][0]:xyLL[area][0]+npts],lat[xyLL[area][1]:xyLL[area][1]+npts],\
                               zc,topo[xyLL[area][1]:xyLL[area][1]+npts,xyLL[area][0]:xyLL[area][0]+npts],\
                               varData[tt,:,xyLL[area][1]:xyLL[area][1]+npts,xyLL[area][0]:xyLL[area][0]+npts],\
                               step,'',area,'')

      #plt.tight_layout()
      plt.suptitle(f'{case} [{hhmm}]',fontsize=12)

      fig.subplots_adjust(left=0.1,right=0.8,bottom=0.2)
      ax_cb = fig.add_axes([0.82, 0.2, 0.03, 0.65])
      cbar=plt.colorbar(cs,cax=ax_cb, ticks=pltCfg['sprec'][2])
      cbar.set_label(pltCfg['sprec'][1],fontsize=10)
      ax_cb_topo = fig.add_axes([0.15, 0.15, 0.65, 0.03])
      cbar=plt.colorbar(cstopo,cax=ax_cb_topo,orientation='horizontal')
      cbar.set_label('Elevation (m)',fontsize=10)
      qk=plt.quiverkey(q1,0.85,0.9, 8, '8 m/s (BS5)', labelpos='E',color='r',coordinates='figure',fontproperties={'size':8},zorder=6)
      #plt.savefig('../figures/sfcWind.sprec.%03d.png'%tt,dpi=300)
      plt.savefig('../figures/%s_sfcWind.%03d.png'%(case,tt),dpi=300)


