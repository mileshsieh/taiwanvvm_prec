import numpy as np
from matplotlib import pyplot as plt
import pickle
import vvmDataUtil as vvm
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
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
    cs=ax.contourf(lon,lat,varData[0],levels=rain_levels,colors=rain_colors,extend='max',alpha=0.6)
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
  case='tpe_20100717_at017'
  zc=np.load('../data/zc.npy')
  lon=np.load('../data/lon.npy')
  lat=np.load('../data/lat.npy')
  topo=np.load('../data/topo.npy')
  print('topo:',topo.min(),topo.max())
  print('lat:',lat.min(),lat.max())
  print('lon:',lon.min(),lon.max())
  npts=256
  area='sw'
  slon=120.40333
  slat=23.45444
  slon=120.543
  slat=23.362
  xCY=np.argmin(np.abs(lon-slon))
  yCY=np.argmin(np.abs(lat-slat))

  dt=10 #min
  stime=np.datetime64('%s-%s-%s 00:00:00'%(case[-14:-10],case[-10:-8],case[-8:-6]))
  tmList=[stime+np.timedelta64(dt*tt,'m') for tt in range(145)]
  #cases
  caseList=pd.read_csv('../data/caseList.94.txt',names=['caseName']).caseName.values
  caseList=['tpe_20180828_sw004',]
  caseList=['tpe_20100717_at017','tpe_20180828_sw004']
  for case in caseList:
    trueData=np.load(f'../data/{case}.sfc.npy')[:,0,:,:]
    dataDict={}
    for hr in [1,3,6]:
      dataDict[f'{hr}hr']=np.load(f'../data/ae_data/{case}.conv.{hr}hr.npy')
  
    tstart=48
    tend=120
    step=15
    #for tt in [93]:
    for tt in range(tstart,tend+1):
      hhmm=pd.to_datetime(str(tmList[tt])).strftime('%H:%M')
      print(case,hhmm)
  
      plt.close()
      fig=plt.figure(figsize=(10,6))
      gs=GridSpec(2,3,figure=fig)
  
      axTS=fig.add_subplot(gs[0,:])
      axTS.plot(tmList[tstart:tend+1],trueData[tstart:tend+1,yCY,xCY],label=f'VVM',color='k')
      for icol,hr in zip(range(3),[1,3,6]):
        print(icol,hr)
        axTS.plot(tmList[tstart:tend+1],dataDict[f'{hr}hr'][tstart:tend+1,0,yCY,xCY],label=f'{hr}hr')
        axTS.xaxis.set_major_locator(mdates.HourLocator())
        axTS.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        axTS.set_xlim(tmList[tstart],tmList[tend+1])
        #axTS.set_ylim(0,85)
        axTS.axvline(tmList[tt],0,1,color='r',lw=2)
        ax=fig.add_subplot(gs[1,icol])
        cs,cstopo,q1=plotSfc(ax,lon[xyLL[area][0]:xyLL[area][0]+npts],lat[xyLL[area][1]:xyLL[area][1]+npts],\
                             zc,topo[xyLL[area][1]:xyLL[area][1]+npts,xyLL[area][0]:xyLL[area][0]+npts],\
                             dataDict[f'{hr}hr'][tt,:,xyLL[area][1]:xyLL[area][1]+npts,xyLL[area][0]:xyLL[area][0]+npts],\
                             step,'','',f'conv:{hr}hr')
        #CY airport
        ax.plot(slon,slat,'rs',markeredgecolor='k')
  
      #plt.tight_layout()
      axTS.grid()
      axTS.set_xlim(tmList[tstart],tmList[tend])
      axTS.set_ylabel('mm/hr')
      axTS.legend()
      fig.subplots_adjust(left=0.08,right=0.85)
      ax_cb = fig.add_axes([0.87, 0.1, 0.03, 0.75])
      cbar=plt.colorbar(cs,cax=ax_cb, ticks=pltCfg['sprec'][2])
      cbar.set_label(pltCfg['sprec'][1],fontsize=15)
      #ax_cb_topo = fig.add_axes([0.15, 0.07, 0.65, 0.03])
      #cbar=plt.colorbar(cstopo,cax=ax_cb_topo,orientation='horizontal')
      #cbar.set_label('Elevation (m)',fontsize=15)
      qk=plt.quiverkey(q1,0.9,0.9, 10, '10 m/s', labelpos='E',coordinates='figure',fontproperties={'size':15},zorder=6)
      plt.suptitle(f'{case} {hhmm}',fontsize=15)
      #plt.savefig('../figures/sfcWind.sprec.%03d.png'%tt,dpi=300)
      plt.savefig('../figures/point/%s.sfc.%s.%03d.png'%(case,area,tt),dpi=300)
  

