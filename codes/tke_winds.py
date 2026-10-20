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
        'TKE':['TKE','TKE ($m^2/s^2$)',np.linspace(0,100,11),'max'],
        'CAPE':['CAPE','CAPE(J/kg)',np.linspace(0.0,3500.0,71,endpoint=True),'max'],
        }

def plotSfc(ax,lon,lat,zc,topo,varData,tke_sfc,step,title,rtitle,ltitle,isTopo=True):
    #ny,nx=topo.shape
    #xx,yy=np.meshgrid(lon,lat)
    rain_thd=float(rtitle.split('>=')[-1])
    if isTopo:
        cstopo=ax.contourf(lon,lat,zc[topo.astype(np.int32)],levels=np.linspace(0,1500,11),cmap='Greys',extend='max')
    #prec
    cs_prec=ax.contour(lon,lat,varData[0],levels=[rain_thd,],colors=['k',],extend='max',alpha=0.6)
    #tke
    cs=ax.contourf(lon,lat,tke_sfc,levels=pltCfg['TKE'][2],cmap='GnBu',extend=pltCfg['TKE'][3])
    #wind (only G6)
    ws=np.sqrt(varData[1,::step,::step]**2+varData[2,::step,::step]**2)
    mask_G6=ws >= 8.0
    lon2D, lat2D = np.meshgrid(lon[::step],lat[::step])
    uG6=varData[1,::step,::step]
    vG6=varData[2,::step,::step]
    q1=ax.quiver(lon2D[mask_G6],lat2D[mask_G6],uG6[mask_G6],vG6[mask_G6],\
            scale=120,scale_units='width',angles='xy',zorder=4,width=0.004,color='r')
    #q1=ax.quiver(lon[::step],lat[::step],varData[1,::step,::step],varData[2,::step,::step],\
    #        scale=120,scale_units='width',angles='xy',zorder=4,color='k',width=0.004,)
    ax.contour(lon,lat,topo,levels=[0.5,10.0],colors=['k','darkgrey'],linewidths=[1.0,0.5],zorder=6)
    ax.set_xlim(lon[0],lon[-1])
    ax.set_ylim(lat[0],lat[-1])
    #ax.xaxis.set_major_locator(ticker.NullLocator())
    #ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.set_title(rtitle,fontsize=12,loc='right')
    ax.set_title(ltitle,fontsize=12,loc='left')
    ax.set_title(title,fontsize=18,loc='center')
    #ax.grid()
    ax.set_aspect('equal')
    if isTopo:
        return cs,cstopo,q1
    else:
        return cs,cs_prec,q1

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
    tke=np.load(f'/data/mileshsieh/turbulence_map/data/processed/tke/{case}.tke.t{tt:03}.npy')
    ny,nx=topo.shape
    x2d,y2d=np.meshgrid(np.arange(nx),np.arange(ny))
    tke_sfc=tke[topo.astype(int),y2d,x2d]
    a=stime+np.timedelta64(dt*tt,'m')
    hhmm=pd.to_datetime(str(a)).strftime('%H:%M')
    print(hhmm)

    plt.close()
    fig=plt.figure(figsize=(len(areaList)*3+1,4))
    #grid=AxesGrid(fig,111,nrows_ncols=(1,3),axes_pad=(0.1,0.4),share_all=False)

    for i,area in enumerate(areaList):
      print(i,area)
      ax=plt.subplot(1,len(areaList),i+1)
      cs,cs_prec,q1=plotSfc(ax,lon[xyLL[area][0]:xyLL[area][0]+npts],lat[xyLL[area][1]:xyLL[area][1]+npts],\
                               zc,topo[xyLL[area][1]:xyLL[area][1]+npts,xyLL[area][0]:xyLL[area][0]+npts],\
                               varData[tt,:,xyLL[area][1]:xyLL[area][1]+npts,xyLL[area][0]:xyLL[area][0]+npts],\
                               tke_sfc[xyLL[area][1]:xyLL[area][1]+npts,xyLL[area][0]:xyLL[area][0]+npts],\
                               step,'','Prec>=20',area,isTopo=False)

      #plt.tight_layout()
      plt.suptitle(f'{case} [{hhmm}]',fontsize=12)

      fig.subplots_adjust(left=0.05,right=0.85,bottom=0.1)
      ax_cb = fig.add_axes([0.87, 0.1, 0.03, 0.75])
      cbar=plt.colorbar(cs,cax=ax_cb, ticks=pltCfg['TKE'][2])
      cbar.set_label(pltCfg['TKE'][1],fontsize=12)
      #ax_cb_topo = fig.add_axes([0.15, 0.15, 0.65, 0.03])
      #cbar=plt.colorbar(cs_prec,cax=ax_cb_topo,orientation='horizontal', ticks=pltCfg['sprec'][2])
      #cbar.set_label(pltCfg['sprec'][1],fontsize=10)
      qk=plt.quiverkey(q1,0.83,0.9, 8, '8 m/s (BS5)', labelpos='E',color='r', coordinates='figure',fontproperties={'size':12},zorder=6)
      #plt.savefig('../figures/sfcWind.sprec.%03d.png'%tt,dpi=300)
      plt.savefig('../figures/%s_tke_sfcWind.%03d.png'%(case,tt),dpi=300)


