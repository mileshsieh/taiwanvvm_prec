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
    #ws=np.sqrt(varData[1,:,:]**2+varData[2,:,:]**2)
    cstopo=ax.contourf(lon,lat,zc[topo.astype(np.int32)],levels=np.linspace(0,1500,11),cmap='Greys',extend='max')
    cs=ax.contourf(lon,lat,varData[0],levels=rain_levels,colors=rain_colors,alpha=0.6)
    q1=ax.quiver(lon[::step],lat[::step],varData[1,::step,::step],varData[2,::step,::step],\
                 scale=180,scale_units='width',angles='xy',zorder=4,color='k')
    ax.contour(lon,lat,topo,levels=[0.5,],colors=['k',],zorder=6)
    ax.set_xlim(lon[0],lon[-1])
    ax.set_ylim(lat[0],lat[-1])
    #ax.xaxis.set_major_locator(ticker.NullLocator())
    #ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.set_title(rtitle,fontsize=10,loc='right')
    ax.set_title(ltitle,fontsize=10,loc='left')
    ax.set_title(title,fontsize=18,loc='center')
    #ax.grid()
    return cs,cstopo,q1    

def plotBoxes(zc,topo,xyLL,npts):
    plt.close()
    cstopo=plt.contourf(zc[topo.astype(np.int32)],levels=np.linspace(0,1500,11),cmap='Greys',extend='max')
    plt.contour(topo,levels=[0.5,],colors=['k',],zorder=6)
    for a in xyLL.keys():
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
        }

if __name__ == "__main__":
  case='tpe_20100802_at001'
  case='tpe_20100717_at017'
  dataDir=vvm.vvmDir+'/%s'%case

  nt=vvm.getNT(dataDir)
  (nz,ny,nx),zc=vvm.getDimension(case,dataDir)
  topo,utopo,vtopo,lon,lat,lon_edge,lat_edge=vvm.getTOPO(dataDir)
  print('caseName=%s'%case)
  print('Dimensions: nx=%d, ny=%d, nz=%d'%(nx,ny,nz))
  print('topo:',topo.min(),topo.max())
  print('lat:',lat.min(),lat.max())
  print('lon:',lon.min(),lon.max())
  npts=256
  #plotBoxes(zc,topo,xyLL,npts)
  #np.save('../data/topo.npy',topo.filled(0.0))
  #np.save('../data/lon.npy',lon.filled(0.0))
  #np.save('../data/lat.npy',lat.filled(0.0))

  dt=10 #min
  stime=np.datetime64('%s-%s-%s 00:00:00'%(case[-14:-10],case[-10:-8],case[-8:-6]))

  #cases
  #caseList=['tpe_20130807_at009','tpe_20130705_at008']
  #caseList=['sw_20190815_sw005','sw_20160902_sw008_n6uvtq']
  caseList=['tpe_20130807_at009','tpe_20100717_at017']
  dataDict={}
  for c in caseList:
    for hr in [1,3,6]:
      dataDict[f'{c}_{hr}hr']=np.load(f'../data/ae_data/{c}.conv.{hr}hr.npy')

  tstart=48
  tend=120
  step=15
  #for tt in [81]:
  for tt in range(tstart,tend+1):
    #for area in xyLL.keys():
    for area in ['sw',]:
      a=stime+np.timedelta64(dt*tt,'m')
      hhmm=pd.to_datetime(str(a)).strftime('%H:%M')
      print(hhmm)

      plt.close()
      fig=plt.figure(figsize=(10,8))
      grid=AxesGrid(fig,111,nrows_ncols=(len(caseList),3),axes_pad=(0.1,0.4),share_all=True)

      for irow,c in enumerate(caseList):
        for icol,hr in zip(range(3),[1,3,6]):
          print(c,icol)
          cs,cstopo,q1=plotSfc(grid[irow*3+icol],lon[xyLL[area][0]:xyLL[area][0]+npts],lat[xyLL[area][1]:xyLL[area][1]+npts],\
                               zc,topo[xyLL[area][1]:xyLL[area][1]+npts,xyLL[area][0]:xyLL[area][0]+npts],\
                               dataDict[f'{c}_{hr}hr'][tt,:,xyLL[area][1]:xyLL[area][1]+npts,xyLL[area][0]:xyLL[area][0]+npts],\
                               step,'',f'conv\n{hr}hr',c)

      #plt.tight_layout()
      plt.suptitle(hhmm,fontsize=20)

      fig.subplots_adjust(left=0.05,right=0.85)
      ax_cb = fig.add_axes([0.87, 0.17, 0.03, 0.65])
      cbar=plt.colorbar(cs,cax=ax_cb, ticks=pltCfg['sprec'][2])
      cbar.set_label(pltCfg['sprec'][1],fontsize=15)
      ax_cb_topo = fig.add_axes([0.15, 0.07, 0.65, 0.03])
      cbar=plt.colorbar(cstopo,cax=ax_cb_topo,orientation='horizontal')
      cbar.set_label('Elevation (m)',fontsize=15)
      qk=plt.quiverkey(q1,0.9,0.9, 10, '10 m/s', labelpos='E',coordinates='figure',fontproperties={'size':15},zorder=6)
      plt.suptitle(hhmm,fontsize=20)
      #plt.savefig('../figures/sfcWind.sprec.%03d.png'%tt,dpi=300)
      plt.savefig('../figures/%s_sfcWind.sprec.%s.%03d.png'%(area,'_'.join(caseList),tt),dpi=300)


