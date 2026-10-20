#!/home/mileshsieh/anaconda3/bin/python
import numpy as np
import vvmDataUtil as vvm
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib,glob,pickle
matplotlib.rc('xtick',labelsize=30)
matplotlib.rc('ytick',labelsize=30)
with open('PrecipitationColorTable.pickle', 'rb') as fc:
  rain_levels, rain_colors = pickle.load(fc)


pltCfg={
  'iTKE':['iTKE','integrated TKE ($kJ/m^2$)',np.linspace(10,170,17),'max'],
  'TKE':['TKE','TKE ($m^2/s^2$)',np.linspace(10,170,17),'max'],
        }
Cp=1004.0
g=9.8
Lv=2264760.0
R=8.314

if __name__ == "__main__":
  dt=10 #min
  var='TKE'
  ztop=4000
  caseList=[a.split('/')[-1].split('.')[0] for a in glob.glob('/data/mileshsieh/turbulence_map/data/processed/tke/*.tke.npy')]
  #for case in caseList:
  #for case in ['tpe20110702cln',]:
  for case in ['lc_20220625',]:
    # case: 20220625
    homeDir='/data2/VVM/taiwanvvm_lc'
    case='lc_20220625'
    dt=10 #min
    ymd=case.split('_')[1]
    stime=np.datetime64('%s-%s-%s 00:00:00'%(ymd[:4],ymd[4:6],ymd[6:]))

    #get data
    dataDir=homeDir+'/%s'%case
    nt=vvm.getNT(dataDir)
    (nz,ny,nx),zc=vvm.getDimension(case,dataDir)
    tp,utp,vtp,lon,lat,lon_edge,lat_edge=vvm.getTOPO(dataDir)

    print('caseName=%s'%case)
    print('Dimensions: nx=%d, ny=%d, nz=%d, nt=%d'%(nx,ny,nz,nt))
    zz=vvm.getFort98Info(dataDir,case,'ZZ')
    topo=zz[tp.astype(int)]

    sprec=np.load('../data/%s.sfc.npy'%case)[:,0,:,:]
    #sprec[np.isnan(sprec)]=0.0
    sprec=np.where(sprec>=1,1.0,0.0)
    x2d,y2d=np.meshgrid(np.arange(nx),np.arange(ny))
    idx_topo=tp.astype(int)

    for tt in [93,]:    
    #for tt in range(36,nt):
      u_intp,v_intp,w_intp=vvm.get3DWind_intp(dataDir,case,tp,utp,vtp,tt)
      uMean=np.nanmean(u_intp)
      vMean=np.nanmean(v_intp)
      #vMean=np.nanmean(np.nanmean(v,axis=-1),axis=-1)
      tke_tt=np.load('/data/mileshsieh/turbulence_map/data/processed/tke/%s.tke,t%03d.npy'%(case,tt))
      tke_500=tke_tt[idx_topo,y2d,x2d]
      #for plotting
      a=stime+np.timedelta64(10*tt,'m')
      hhmm=pd.to_datetime(str(a)).strftime('%H:%M')      
      print(case,hhmm,uMean,vMean)
      plt.close()
      fig = plt.figure(figsize=(12,16))
      ax = fig.add_axes([0.1,0.1,0.7,0.7])
      # draw filled contours.
      plt.contour(topo,levels=[0.5,],colors=['k',],linewidths=[3,])
      plt.contourf(topo,cmap='Greys')
      cs = ax.contourf(tke_500,levels=pltCfg[var][2],colors=rain_colors[1:],extend=pltCfg[var][3],zorder=3,alpha=0.6)
      #cs2 = ax.contourf(sprec[tt-36,:,:],levels=[20,2000],colors='k',hatches=['.',])
      cs2 = ax.contour(sprec[tt,:,:],levels=[0.5,],colors='k',linewidths=3.0,zorder=5)
      plt.xlim(250,750)
      plt.ylim(120,920)
      plt.gca().set_xticks([])
      plt.gca().set_yticks([])

      axin=plt.gca().inset_axes([0.01, 0.84, 0.20, 0.15])
      #q=axin.quiver(np.zeros(idx_ztop),zc[:idx_ztop],uMean,vMean,scale=50)
      q=axin.barbs(0.5,0.5,uMean,vMean,length=8, pivot='middle',lw=2,barb_increments={'half':0.5,'full':1.0,'flag':5.0})
      axin.set_xticks([])
      axin.set_yticks([])
      #axin.grid()
      #axin.set_ylim(0,4000)
      #axin.set_ylabel('Height(m)',fontsize=40)
      #qk = axin.quiverkey(q, 0.2, 0.8, 5, '5 m/s', labelpos='E',coordinates='axes',fontproperties={'size':40})
       # color bar of var
      cax_sprec=fig.add_axes([0.85, 0.1, 0.03, 0.7])
      cbar_lwp=plt.colorbar(cs, cax=cax_sprec,ticks=pltCfg[var][2])
      cbar_lwp.set_label(pltCfg[var][1],fontsize=30)

      # add title
      ax.set_title(pltCfg[var][0],fontsize=30)
      plt.suptitle('Local Time=%s [case:%s]'%(hhmm,case),fontsize=30)
      plt.savefig('../figures/TKE/%s.%s.%03d.png'%(case,var,tt))
      
