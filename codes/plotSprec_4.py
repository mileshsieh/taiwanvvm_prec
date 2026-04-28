import numpy as np
import vvmDataUtil as vvm
from matplotlib import pyplot as plt
import pickle
import pandas as pd
from netCDF4 import Dataset as netcdf
import matplotlib
matplotlib.rc('xtick',labelsize=15)
matplotlib.rc('ytick',labelsize=15)

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

if __name__ == "__main__":
  case='tpe_20100802_at001'
  dataDir=vvm.vvmDir+'/%s'%case

  valley_box={'ts':[121.4241,121.5031,25.08828,25.15803],
              'kl':[121.5682,121.7542,25.03248,25.17198]}
  plonlat={'Taipei':[121.2,122.0,24.8,25.4],
           'SWCoast':[119.9,120.7,22.4,23.4]}

  nt=vvm.getNT(dataDir)
  (nz,ny,nx),zc=vvm.getDimension(case,dataDir)
  topo,utopo,vtopo,lon,lat,lon_edge,lat_edge=vvm.getTOPO(dataDir)

  print('caseName=%s'%case)
  print('Dimensions: nx=%d, ny=%d, nz=%d'%(nx,ny,nz))
  print('topo:',topo.min(),topo.max())
  print('lat:',lat.min(),lat.max())
  print('lon:',lon.min(),lon.max())
  dt=10 #min
  stime=np.datetime64('%s-%s-%s 00:00:00'%(case[-14:-10],case[-10:-8],case[-8:-6]))
  step=6
  #no rain cases
  caseList=['tpe_20100802_at001','tpe_20140810_sw001','sw_20140810_sw001','sw_20140810_sw001_nudging6hr']
  #comparison
  caseDict={'tpe20120715cln':'Double Peaks\n(Intensified)',
            'tpe20140825dty':'Single Peak',
            'tpe20090827dtyP3K':'Double Peaks\n(Comparable)',
            'tpe20110816dtyP3K':'No Precipitation',
           }
  area='SWCoast'
  #for tt in [81]:
  #for tt in range(36,121):
  for tt in range(12,121):
    a=stime+np.timedelta64(dt*tt,'m')
    hhmm=pd.to_datetime(str(a)).strftime('%H:%M')
    print(hhmm)

    plt.close()
    fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=(16,20))
    
    #for case,ax in zip(caseDict.keys(),[ax1,ax2,ax3,ax4]):
    for case,ax in zip(caseList,[ax1,ax2,ax3,ax4]):
      dataDir=vvm.vvmDir+'/%s'%case

      sprec=vvm.getData(dataDir,case,'sprec',tt)
      usfc,vsfc=vvm.getSfcWind_intp(dataDir,case,topo,utopo,vtopo,tt)
      cstopo=ax.contourf(lon,lat,zc[topo.astype(np.int)],levels=np.linspace(0,1500,11),cmap='Greys',extend='max')
      cs=ax.contourf(lon,lat,sprec,levels=rain_levels,colors=rain_colors,alpha=0.6)
      q1=ax.quiver(lon[::step],lat[::step],usfc[::step,::step],vsfc[::step,::step],\
                   scale=180,scale_units='width',angles='xy',zorder=4,color='k')

      #for area in areas.keys():
      #  ystart,yend,xstart,xend=areas[area]
      #  plt.plot(lon[[xstart,xstart,xend,xend,xstart]],lat[[ystart,yend,yend,ystart,ystart]],color='red',lw=2)
      ax.contour(lon,lat,topo,levels=[0.5,],colors=['k',],zorder=6)
      ax.contour(lon,lat,topo,levels=[2.5],colors=['gray'],zorder=6)
      ax.set_xlim(plonlat[area][0],plonlat[area][1])
      ax.set_ylim(plonlat[area][2],plonlat[area][3])
      ax.set_title(case,fontsize=15,loc='right')
      #ax.set_title(caseDict[case],fontsize=25,loc='left')

      #ax.contour(lon,lat,reg3,levels=[0.5],zorder=7,linestyles='dashed',colors='darkred')
      #for valley in valley_box.keys():
      #  boxx = [valley_box[valley][0],valley_box[valley][0],valley_box[valley][1],valley_box[valley][1],valley_box[valley][0]]
      #  boxy = [valley_box[valley][2],valley_box[valley][3],valley_box[valley][3],valley_box[valley][2],valley_box[valley][2]]
      #  ax.plot(boxx, boxy, color='r', lw=2)

      ax.plot(121.5395,25.0144,marker='*',markersize=15,mfc='yellow',mec='k')

    fig.subplots_adjust(right=0.8)
    ax_cb = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    cbar=plt.colorbar(cs,cax=ax_cb, ticks=pltCfg['sprec'][2])
    cbar.set_label(pltCfg['sprec'][1],fontsize=30)
    qk = plt.quiverkey(q1,0.2,0.05, 5, '5 m/s', labelpos='E',coordinates='figure',fontproperties={'size':20},zorder=6)
    plt.suptitle(hhmm,fontsize=20)
    #plt.savefig('../figures/sfcWind.sprec.%03d.png'%tt,dpi=300)
    plt.savefig('../figures/%s_sfcWind.sprec.%s.%03d.png'%(area,'_'.join(caseList),tt),dpi=300)


