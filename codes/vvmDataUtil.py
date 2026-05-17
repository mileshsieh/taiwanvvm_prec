import matplotlib
import numpy as np
import pandas as pd
import datetime,matplotlib
from netCDF4 import Dataset as netcdf       # netcdf4-python module
from netCDF4 import num2date
import sys,os,glob,time
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
Cp=1004.0
g=9.8
Lv=2264760.0
R=8.314
vvmDir='/data2/VVM/taiwanvvm_summer'

varFn={'qc':'%s.L.Thermodynamic-%06d.nc',
       'qr':'%s.L.Thermodynamic-%06d.nc',
       'qv':'%s.L.Thermodynamic-%06d.nc',
       'qi':'%s.L.Thermodynamic-%06d.nc',
       'th':'%s.L.Thermodynamic-%06d.nc',
       'qrim':'%s.L.Thermodynamic-%06d.nc',
       'ni':'%s.L.Thermodynamic-%06d.nc',
       'u':'%s.L.Dynamic-%06d.nc',
       'v':'%s.L.Dynamic-%06d.nc',
       'w':'%s.L.Dynamic-%06d.nc',
       'tr01':'%s.L.Tracer-%06d.nc',
       'tr02':'%s.L.Tracer-%06d.nc',
       'tr03':'%s.L.Tracer-%06d.nc',
       'tr04':'%s.L.Tracer-%06d.nc',
       'tr05':'%s.L.Tracer-%06d.nc',
       'tr06':'%s.L.Tracer-%06d.nc',
       'tr07':'%s.L.Tracer-%06d.nc',
       'tr08':'%s.L.Tracer-%06d.nc',
       'tr09':'%s.L.Tracer-%06d.nc',
       'tr10':'%s.L.Tracer-%06d.nc',
       'tr11':'%s.L.Tracer-%06d.nc',
       'tr12':'%s.L.Tracer-%06d.nc',
       'tr13':'%s.L.Tracer-%06d.nc',
       'tr14':'%s.L.Tracer-%06d.nc',
       'tr15':'%s.L.Tracer-%06d.nc',
       'tr16':'%s.L.Tracer-%06d.nc',
       'tr17':'%s.L.Tracer-%06d.nc',
       'tr18':'%s.L.Tracer-%06d.nc',
        }
windIntp={'u':2,'v':1}

def _getData_fromNC_3D(dataDir,case,var,tt):
  nc_f=dataDir+'/archive/'+varFn[var]%(case,tt)
  nc_fid = netcdf(nc_f, 'r')
  varData=nc_fid.variables[var][0,:,:,:]
  nc_fid.close()
  return varData

def _getData_fromNC_SFC(dataDir,case,var,tt):
  nc_f=dataDir+'/archive/%s.C.Surface-%06d.nc'%(case,tt)
  nc_fid = netcdf(nc_f, 'r')
  varData=nc_fid.variables[var][0,:,:]
  nc_fid.close()
  return varData

def getDataCutting(dataDir,case,var,tt,istart,iend,jstart,jend):
  if var in varFn.keys():
  #if var in ['qc','qv','qr','qi','th','qrim','ni','u','v','w','tr01','tr02','tr03','tr04','tr05','tr06']:
    varData=_getData_fromNC_3D(dataDir,case,var,tt)
    varDataCutting=varData[:,jstart:jend,istart:iend]
  elif var in ['sprec',]:
    varData=_getData_fromNC_SFC(dataDir,case,var,tt)
    varDataCutting=varData[jstart:jend,istart:iend]*3600
  elif var in ['tg',]:
    varData=_getData_fromNC_SFC(dataDir,case,var,tt)
    varDataCutting=varData[jstart:jend,istart:iend]
  else:
    return None
  return varDataCutting

def getData(dataDir,case,var,tt):
  if var in varFn.keys():
    varData=_getData_fromNC_3D(dataDir,case,var,tt)
    #if var in ['u','v']:
    #  varData=_getData_fromNC_SFC(dataDir,case,var,tt)
    #  varData=0.5*(varData+np.roll(varData,-1,axis=windIntp[var]))
      
  elif var in ['sprec']:
    #print('getData',dataDir,case,var,tt)
    varData=_getData_fromNC_SFC(dataDir,case,var,tt)*3600
  elif var in ['tg']:
    #print('getData',dataDir,case,var,tt)
    varData=_getData_fromNC_SFC(dataDir,case,var,tt)
  elif var in ['thg']:
    #print('getData',dataDir,case,var,tt)
    varData=_getData_fromNC_SFC(dataDir,case,'tg',tt)
    pi=getFort98Info(dataDir,case,'PIBAR')[:-1].reshape((-1,1,1))
    varData=varData/pi
    
  else:
    return None
  return varData

def getZC(dataDir,case):
  #nc_f=dataDir+'/archive/%s.L.Dynamic-000000.nc'%case
  nc_f=dataDir+'/archive/%s.L.Thermodynamic-000000.nc'%case
  nc_fid = netcdf(nc_f, 'r')
  zc = nc_fid.variables['zc'][:]
  nc_fid.close()
  return zc

def getTOPO(dataDir):
  nc_f=dataDir+'/TOPO.nc'
  nc_fid = netcdf(nc_f, 'r')
  topo=nc_fid.variables['topo'][:,:]
  lat=nc_fid.variables['lat'][:]
  lon=nc_fid.variables['lon'][:]
  nc_fid.close()
  ny,nx=topo.shape
  utopo=np.array([np.maximum(topo[j,:],np.roll(topo[j,:],-1))for j in range(ny)])
  vtopo=np.array([np.maximum(topo[:,i],np.roll(topo[:,i],-1))for i in range(nx)]).T
  late=0.5*(lat+np.roll(lat,-1))
  late[-1]=lat[-1]
  lone=0.5*(lon+np.roll(lon,-1))
  lone[-1]=lon[-1]
  return topo,utopo,vtopo,lon,lat,lone,late

def getNT(dataDir):
  fList=glob.glob(dataDir+'/archive/*Thermodynamic*.nc')
  nt=len(fList)
  return nt

def getDimension(case,dataDir):
  nc_f=dataDir+'/archive/%s.L.Thermodynamic-%06d.nc'%(case,0)
  #print(nc_f)
  nc_fid = netcdf(nc_f, 'r')
  varData=nc_fid.variables['qv'][0,:,:,:]
  zc=nc_fid.variables['zc'][:]
  nc_fid.close()
  return varData.shape,zc

def calcEs(temp_in_C):
  #return 6.11*np.exp(19.85-(5418.7/(temp_in_C+273.15)))
  return 611.2*np.exp((17.67*temp_in_C)/(temp_in_C+243.5))
def calcQs(temp_in_C,pres_in_Pa):
  es=calcEs(temp_in_C)
  rs=0.622*es/(pres_in_Pa-es)
  return rs/(1.0+rs)

def calc_gamma_m(qvs_in_kgkg,T_in_K):
    return (g*(1+(2500000*qvs_in_kgkg)/(287*T_in_K)))/(Cp*(1+(2500000*2500000*qvs_in_kgkg)/(1004*461*T_in_K*T_in_K)))

def getCircMsk(rad,xidx,yidx,nx,ny,dx=0.5): #units of rad and dx is km
  xx,yy=np.meshgrid(np.arange(nx),np.arange(ny))
  distance=np.sqrt((xx-xidx)*(xx-xidx)+(yy-yidx)*(yy-yidx))*dx
  dd=np.where(distance<=rad,1,0)
  yin,xin=np.where(dd>=1)
  return dd[yin.min():yin.max(),xin.min():xin.max()],yin.min(),yin.max(),xin.min(),xin.max()

def calcBuoyancy(dataDir,case,xidx,yidx,tt,topo,zc,msk,ystart,yend,xstart,xend):
  idx_topo=topo[ystart:yend,xstart:xend].astype(np.int)
  nz=zc.shape[0]
  #get th
  th=getData(dataDir,case,'th',tt)[:,ystart:yend,xstart:xend]
  #get qv(nz,ny,nx) and qv_sfc
  qv=getData(dataDir,case,'qv',tt)[:,ystart:yend,xstart:xend]
  thv_hotspot=th[:,yidx-ystart,xidx-xstart]*(1+0.61*qv[:,yidx-ystart,xidx-xstart])
  for k in range(nz):
    th[k,:,:]=np.where(idx_topo>k,np.nan,th[k,:,:])
    qv[k,:,:]=np.where(idx_topo>k,np.nan,qv[k,:,:])

  thv_mean=np.nanmean((th.reshape((nz,-1))*(1+0.61*qv.reshape((nz,-1))))[:,msk.flatten()],axis=-1)
  buoyancy_profile=g*(thv_hotspot-thv_mean)/thv_mean
  
  return buoyancy_profile

def calcBuoyancySqare(dataDir,case,tt,yy,xx,topo,zc,h_wd=5):
  ystart=yy-h_wd
  yend=yy+h_wd
  xstart=xx-h_wd
  xend=xx+h_wd 
  idx_topo=topo[ystart:yend,xstart:xend].astype(np.int)
  ny=yend-ystart
  nx=xend-xstart
  x2d,y2d=np.meshgrid(np.arange(nx),np.arange(ny))
  nz=zc.shape[0]
  #get th
  th=getData(dataDir,case,'th',tt)[:,ystart:yend,xstart:xend]
  #get qv(nz,ny,nx) and qv_sfc
  qv=getData(dataDir,case,'qv',tt)[:,ystart:yend,xstart:xend]
  for k in range(nz):
    th[k,:,:]=np.where(idx_topo>k,np.nan,th[k,:,:])
    qv[k,:,:]=np.where(idx_topo>k,np.nan,qv[k,:,:])

  thv_mean=np.nanmean(np.nanmean(th*(1+0.61*qv),axis=-1),axis=-1)
  thv_hotspot=th[:,ny//2,nx//2]*(1+0.61*qv[:,ny//2,nx//2])
  buoyancy_profile=g*(thv_hotspot-thv_mean)/thv_mean
  
  return buoyancy_profile

def _calcParcelProfile(dataDir,case,tt,yy,xx,xytopo,zc):
  idx_topo=xytopo.astype(np.int)
  nz=zc.shape[0]
  #get PIBAR, PBAR, RHO
  pbar=getFort98Info(dataDir,case,'PBAR')
  pibar=getFort98Info(dataDir,case,'PIBAR')
  rho=getFort98Info(dataDir,case,'RHO')
  #get temp
  temp=getData(dataDir,case,'th',tt)[:,yy,xx]*pibar
  temp_sfc=temp[idx_topo]
  #create above ground mask and delta z above ground
  del_z=zc[1:]-zc[:-1]
  above_ground=zc>zc[idx_topo] 
  del_z_ground=np.where(above_ground,zc-zc[idx_topo],0.0)
  
  parcel_gamma_d=temp_sfc-g*del_z_ground/Cp
  
  #get qv(nz,ny,nx) and qv_sfc
  qv=getData(dataDir,case,'qv',tt)[:,yy,xx]
  qv_sfc=qv[idx_topo]
  #calc es
  e=np.where(above_ground,qv_sfc*pbar/0.622,0.0)
  # calculate es in Pa
  es=calcEs(parcel_gamma_d-273.15)
  del_e=e-es
  del_qv=0.622*(e-es)/pbar
  #get LCL where del_e >=0
  above_LCL= del_e >= 0
  LCL_idx=np.argmax(above_LCL)
  #calculate parcel lapse rate
  temp_parcel=parcel_gamma_d
  qv_parcel=np.ones(zc.shape[0])*qv_sfc

  for k in range(nz-1):
    qvs=calcQs(temp_parcel[k]-273.15,pbar[k])
    gamma_m=calc_gamma_m(qvs,temp_parcel[k])
    temp_parcel[k+1]=np.where(above_LCL[k],temp_parcel[k]-del_z[k]*gamma_m,temp_parcel[k+1])
    qv_parcel[k]=np.where(above_ground[k],qv_parcel[k],0.0)
    qv_parcel[k]=np.where(above_LCL[k],qvs,qv_parcel[k])
    #print(k,temp[k],temp_parcel[k],qv_parcel[k],qvs,(temp_parcel[k+1]-temp_parcel[k])/(zc[k+1]-zc[k]))
    #if (k > 15)and(k<69): 

  #calculate CAPE
  temp_lfc=np.where(above_LCL,temp,temp_parcel+5)
  above_LFC= temp_parcel > temp_lfc
  LFC_idx=np.argmax(above_LFC)
  #EL_idx=np.array([np.argmax(aboveLFC_flat[::-1,k],axis=-1) for k in range(aboveLFC_flat.shape[1])])
  #EL_idx=(nz-EL_idx-1).reshape((ny,nx))
  CAPE=np.where(above_LFC[:-1],g*del_z*(temp_parcel[:-1]*(1+0.61*qv_parcel[:-1])-temp[:-1]*(1+0.61*qv[:-1]))/(temp[:-1]*(1+0.61*qv[:-1])),0)
  CAPE=np.sum(CAPE)
  temp=np.where(above_ground,temp,np.nan)
  temp_parcel=np.where(above_ground,temp_parcel,np.nan)
  pCAPE=g*(temp_parcel[:-1]*(1+0.61*qv_parcel[:-1])-temp[:-1]*(1+0.61*qv[:-1]))/(temp[:-1]*(1+0.61*qv[:-1]))
  return temp,temp_parcel,qv_parcel,LCL_idx,LFC_idx,CAPE,pCAPE

def calcCAPE(dataDir,case,tt,topo,zc):
  idx_topo=topo.astype(np.int)
  ny,nx=topo.shape
  x2d,y2d=np.meshgrid(np.arange(nx),np.arange(ny))
  nz=zc.shape[0]
  #get PIBAR, PBAR, RHO
  pbar=getFort98Info(dataDir,case,'PBAR').reshape((-1,1,1))
  pibar=getFort98Info(dataDir,case,'PIBAR').reshape((-1,1,1))
  rho=getFort98Info(dataDir,case,'RHO').reshape((-1,1,1))
  #print(pbar_3D.shape,pibar_3D.shape,rho_3D.shape)

  #get th(nz,ny*nx) and th_sfc(ny,nx)
  temp=getData(dataDir,case,'th',tt)*pibar
  temp_sfc=temp[idx_topo,y2d,x2d]
  qv=getData(dataDir,case,'qv',tt)
  qv_sfc=qv[idx_topo,y2d,x2d]
  for k in range(idx_topo.max()):
    temp[k,:,:]=np.where(idx_topo>k,temp_sfc,temp[k,:,:])

  #create above ground mask and delta z above ground
  temp_upper=temp_sfc
  qv_upper=qv_sfc
  cape=np.zeros((ny,nx))
  for k in range(0,nz-1):
    qvs=calcQs(temp_upper-273.15,pbar[k])
    gamma_m=calc_gamma_m(qvs,temp_upper)
    gamma_d=g/Cp
    dTdz=np.where(idx_topo>k,0.0,np.where(qv_upper>qvs,gamma_m,gamma_d))
    temp_upper=temp_upper-dTdz*(zc[k+1]-zc[k])
    qv_upper=np.where(qv_upper>qvs,qvs,qv_sfc)
    #print(qvs.shape,qv.shape,dTdz.shape,gamma_m.shape,topo.shape)
    #print(k,qvs[512,512],qv_upper[512,512],dTdz[512,512],gamma_m[512,512],topo[512,512])
    tv_parcel=temp_upper*(1+0.61*qv_upper)
    tv_env=temp[k+1,:,:]*(1+0.61*qv[k+1,:,:])
    cape=cape+np.where(tv_parcel>tv_env,g*(zc[k+1]-zc[k])*(tv_parcel-tv_env)/tv_env,0)
    #cape=cape+np.where(temp_upper>temp[k+1,:,:],
    #                   g*(zc[k+1]-zc[k])*(temp_upper-temp[k+1,:,:])/(temp[k+1,:,:]),0)
  print(cape.shape,cape.max(),cape.min(),cape[512,512])
  return cape    

def calcCAPECutting(dataDir,case,tt,topo,zc,xstart,xend,ystart,yend):
  idx_topo=topo[ystart:yend,xstart:xend].astype(np.int)
  ny,nx=topo[ystart:yend,xstart:xend].shape
  x2d,y2d=np.meshgrid(np.arange(nx),np.arange(ny))
  nz=zc.shape[0]
  #get PIBAR, PBAR, RHO
  pbar=getFort98Info(dataDir,case,'PBAR').reshape((-1,1,1))
  pibar=getFort98Info(dataDir,case,'PIBAR').reshape((-1,1,1))
  rho=getFort98Info(dataDir,case,'RHO').reshape((-1,1,1))
  #print(pbar_3D.shape,pibar_3D.shape,rho_3D.shape)

  #get th(nz,ny*nx) and th_sfc(ny,nx)
  temp=getData(dataDir,case,'th',tt)[:,ystart:yend,xstart:xend]*pibar
  temp_sfc=temp[idx_topo,y2d,x2d]
  qv=getData(dataDir,case,'qv',tt)[:,ystart:yend,xstart:xend]
  qv_sfc=qv[idx_topo,y2d,x2d]
  for k in range(idx_topo.max()):
    temp[k,:,:]=np.where(idx_topo>k,temp_sfc,temp[k,:,:])

  #create above ground mask and delta z above ground
  temp_upper=temp_sfc
  qv_upper=qv_sfc
  cape=np.zeros((ny,nx))
  for k in range(0,nz-1):
    qvs=calcQs(temp_upper-273.15,pbar[k])
    gamma_m=calc_gamma_m(qvs,temp_upper)
    gamma_d=g/Cp
    dTdz=np.where(idx_topo>k,0.0,np.where(qv_upper>qvs,gamma_m,gamma_d))
    temp_upper=temp_upper-dTdz*(zc[k+1]-zc[k])
    qv_upper=np.where(qv_upper>qvs,qvs,qv_sfc)
    #print(qvs.shape,qv.shape,dTdz.shape,gamma_m.shape,topo.shape)
    tv_parcel=temp_upper*(1+0.61*qv_upper)
    tv_env=temp[k+1,:,:]*(1+0.61*qv[k+1,:,:])
    cape=cape+np.where(tv_parcel>tv_env,g*(zc[k+1]-zc[k])*(tv_parcel-tv_env)/tv_env,0)
    #cape=cape+np.where(temp_upper>temp[k+1,:,:],
    #                   g*(zc[k+1]-zc[k])*(temp_upper-temp[k+1,:,:])/(temp[k+1,:,:]),0)
  return cape   
 
def calcPBLHonTime(dataDir,case,tt,topo,zc):
  idx_topo=topo.astype(np.int)
  ny,nx=topo.shape
  x2d,y2d=np.meshgrid(np.arange(nx),np.arange(ny))
  nz=zc.shape[0]
  th=getData(dataDir,case,'th',tt)
  th_sfc=th[idx_topo,y2d,x2d]
  for k in range(idx_topo.max()):
    th[k,:,:]=np.where(idx_topo>k,th_sfc,th[k,:,:])

  th_sfc_3D=th_sfc.reshape((-1,ny,nx))+0.5
  overPBL=np.where(th>th_sfc_3D,1,0)
  overPBL_flat=overPBL.reshape((nz,ny*nx))
  PBLH_idx=np.array([np.argmax(overPBL_flat[:,k],axis=-1) for k in range(overPBL_flat.shape[1])])
  print('PBLH idx:',PBLH_idx.shape,PBLH_idx.max(),PBLH_idx.min())
  PBLH=zc[PBLH_idx]
  print('PBLH:',PBLH.shape,PBLH.max(),PBLH.min())
  #test=PBLH-(topo*100.0).reshape((ny*nx))
  #print(np.where(test<0)[0])
  PBLH=PBLH.reshape((ny,nx))-topo*100.0+50.0
  print('PBLH over sfc:',PBLH.shape,PBLH.max(),PBLH.min())
  
  return PBLH

def _calcMLDFromTG_tt(dataDir,case,tt,topo,zc):
  idx_topo=topo.astype(np.int)
  ny,nx=topo.shape
  x2d,y2d=np.meshgrid(np.arange(nx),np.arange(ny))
  nz=zc.shape[0]
  th=getData(dataDir,case,'th',tt)
  th_g=getData(dataDir,case,'thg',tt)
  print(th_g.shape,th.shape)
  for k in range(idx_topo.max()):
    th[k,:,:]=np.where(idx_topo>k,th_g,th[k,:,:])

  th_g_3D=th_g.reshape((-1,ny,nx))
  overMLD=np.where(th>th_g_3D,1,0)
  overMLD_flat=overMLD.reshape((nz,ny*nx))
  MLD_idx=np.array([np.argmax(overMLD_flat[:,k],axis=-1) for k in range(overMLD_flat.shape[1])])
  print('MLD idx:',MLD_idx.shape,MLD_idx.max(),MLD_idx.min())
  MLD=zc[MLD_idx]
  print('MLD:',MLD.shape,MLD.max(),MLD.min())
  #test=MLD-(topo*100.0).reshape((ny*nx))
  #print(np.where(test<0)[0])
  MLD=MLD.reshape((ny,nx))-topo*100.0+50.0
  print('MLD over sfc:',MLD.shape,MLD.max(),MLD.min())
  
  return MLD

def calcCloudAndLWPonTime(case,tt,topo,zc):
  idx_topo=topo.astype(np.int)
  ny,nx=topo.shape
  lwp=np.zeros((ny,nx))
  rho=getFort98Info(dataDir,case,'RHO')[:-1].reshape((-1,1,1))
  nz=zc.shape[0]
  qc=getData(dataDir,case,'qc',tt)
  for k in range(idx_topo.max()):
    qc[k,:,:]=np.where(idx_topo>k,0.0,qc[k,:,:])
  #for LWP
  for k in range(nz-1):
    dz=zc[k+1]-zc[k]
    mean_q=0.5*(qc[k,:,:]+qc[k+1,:,:])
    mean_rho=0.5*(rho[k,:,:]+rho[k+1,:,:])
    #print(tt,k,dz,mean_rho,mean_q.max()*1000)
    lwp=lwp+mean_q*mean_rho*dz*1000
  #for CTH CBH
  cld_2D=np.where(qc>1.0e-5,1,0)
  cld_2D_flat=cld_2D.reshape((nz,ny*nx))
  isCloudy=np.where(np.sum(cld_2D_flat,axis=0)>=1,1,0)
  #cloud top height
  CTH_idx=nz-1-np.array([np.argmax(cld_2D_flat[::-1,k],axis=-1) for k in range(cld_2D_flat.shape[1])])
  print('CTH index:',CTH_idx.shape)
  CTH=zc[CTH_idx]
  print('CTH(all grid):',CTH.shape,CTH.max(),CTH.min())
  #set nan where no cloud
  CTH=np.where(isCloudy>0,CTH,np.nan)
  print('CTH(only cloudy):',CTH.shape,np.nanmax(CTH),np.nanmin(CTH))

  CBH_idx=np.array([np.argmax(cld_2D_flat[:,k],axis=-1) for k in range(cld_2D_flat.shape[1])])
  CBH=zc[CBH_idx]
  print('CBH(all grid):',CBH.shape,CBH.max(),CBH.min())
  #set nan where no cloud
  CBH=np.where(isCloudy>0,CBH,np.nan)
  print('CBH(only cloudy):',CBH.shape,np.nanmax(CBH),np.nanmin(CBH))

  return lwp,CTH.reshape((ny,nx))-(topo*100.0)+50.0,CBH.reshape((ny,nx))-(topo*100.0)+50.0

def getFort98Info(dataDir,case,var):
  #get rho
  fname='%s/fort.98'%dataDir
  #fname='./tpe20060718cln_fort.98'
  #fname='./CWB/fort.98'
  df_k=pd.read_csv(fname,skiprows=262,sep='\s+',nrows=70,
                   names=['K', 'RHO','THBAR','PBAR','PIBAR','QVBAR'])
  rho=df_k[var].values
  #print(rho.shape)
  return rho

def calcTKE(u,v,w):

  tke=0.5*(u*u+v*v+w*w)
  return tke

def calcPseAbd(lwp):
  tau=0.19*np.power(lwp,5.0/6.0)*np.power(50.0,1.0/3.0)
  pseudoAlbedo=tau/(tau+6.8)
  return pseudoAlbedo

def calcThetaV(th,qv,zc):
  thetaV=th*(1+0.61*qv)
  return zc,thetaV


def get3DWind_intp(dataDir,case,tp,utp,vtp,tt):
  topo=tp.copy()
  utopo=utp.copy()
  vtopo=vtp.copy()
  utopo[utopo<1.0]=1.0
  vtopo[vtopo<1.0]=1.0
  topo[topo<1.0]=1.0
  idx_topo=topo.astype(int)
  idx_utopo=utopo.astype(int)
  idx_vtopo=vtopo.astype(int)
  ny,nx=utopo.shape
  x2d,y2d=np.meshgrid(np.arange(nx),np.arange(ny))
  result=np.zeros((ny,nx))
  uData=getData(dataDir,case,'u',tt)
  for k in range(idx_utopo.max()+1):
    uData[k,:,:]=np.where(idx_utopo>k,np.nan,uData[k,:,:])
  uData=np.nanmean([uData,np.roll(uData,1,axis=2)],axis=0)

  vData=getData(dataDir,case,'v',tt)
  for k in range(idx_vtopo.max()+1):
    vData[k,:,:]=np.where(idx_vtopo>k,np.nan,vData[k,:,:])
  vData=np.nanmean([vData,np.roll(vData,1,axis=1)],axis=0)

  #fill nan to 0.0 for valleys with only one grid spacing
  for k in range(idx_topo.max()+1):
    uData[k,:,:]=np.where(idx_topo>k,uData[k,:,:],np.where(np.isnan(uData[k,:,:]),0.0,uData[k,:,:]))
    vData[k,:,:]=np.where(idx_topo>k,vData[k,:,:],np.where(np.isnan(vData[k,:,:]),0.0,vData[k,:,:]))

  return uData,vData


def getSfcWind_intp(dataDir,case,tp,utp,vtp,tt):
  topo=tp.copy()
  utopo=utp.copy()
  vtopo=vtp.copy()
  utopo[utopo<1.0]=1.0
  vtopo[vtopo<1.0]=1.0
  topo[topo<1.0]=1.0
  idx_topo=topo.astype(np.int)
  idx_utopo=utopo.astype(np.int)
  idx_vtopo=vtopo.astype(np.int)
  ny,nx=utopo.shape
  x2d,y2d=np.meshgrid(np.arange(nx),np.arange(ny))
  result=np.zeros((ny,nx))
  uData=getData(dataDir,case,'u',tt)
  for k in range(idx_utopo.max()+1):
    uData[k,:,:]=np.where(idx_utopo>k,np.nan,uData[k,:,:])
  uData=np.nanmean([uData,np.roll(uData,1,axis=2)],axis=0)

  vData=getData(dataDir,case,'v',tt)
  for k in range(idx_vtopo.max()+1):
    vData[k,:,:]=np.where(idx_vtopo>k,np.nan,vData[k,:,:])
  vData=np.nanmean([vData,np.roll(vData,1,axis=1)],axis=0)

  usfc=uData[idx_topo,y2d,x2d]
  vsfc=vData[idx_topo,y2d,x2d]
  usfc[np.isnan(usfc)]=0.0
  vsfc[np.isnan(vsfc)]=0.0
  return usfc,vsfc

def getVarSfcOnTime(dataDir,case,var,topo,tt):
  calcT=False
  idx_topo=topo.astype(np.int)
  ny,nx=topo.shape
  x2d,y2d=np.meshgrid(np.arange(nx),np.arange(ny))
  result=np.zeros((ny,nx))
  if var=='t':
    calcT=True
    pi=getFort98Info(dataDir,case,'PIBAR')[:-1].reshape((-1,1,1))
    print(pi.shape)
    var='th'
  varData=getData(dataDir,case,var,tt)
  #print(varData.shape)
  if calcT:
    tmp=varData*pi
    result=tmp[idx_topo,y2d,x2d]-273.15
    #result[tt,:,:]=varData[idx_topo,y2d,x2d]*pi[idx_topo]
  else:
    result=varData[idx_topo,y2d,x2d]
  #print(tt)
  return result

def plotSprecConfig(case,pic_path,sprec,u,v,topo_int,lat,lon,hhmm,step=5):
  from mpl_toolkits.basemap import Basemap, cm
  import matplotlib

  matplotlib.rc('xtick',labelsize=25)
  matplotlib.rc('ytick',labelsize=25)

  thd=0.01
  topo=topo_int*100.0
  lonLL=lon[0]
  lonUR=lon[-1]
  latLL=lat[0]
  latUR=lat[-1]
  levs=np.linspace(0,3900,14,endpoint=True)
  cm_sprec,levs_sprec=setSprecColorbar(thd)
  print(sprec.shape)
  ny,nx=sprec.shape
  masked_sprec = np.ma.masked_where(sprec<thd, sprec)
  print(hhmm,sprec.max())
  #plot
  plt.close()
  fig = plt.figure(figsize=(16,16))
  ax1 = fig.add_axes([0.1,0.2,0.7,0.7])

  m = Basemap(llcrnrlon=lonLL,llcrnrlat=latLL,urcrnrlon=lonUR,urcrnrlat=latUR,
              projection='cyl',
              resolution ='h')
              #resolution ='i')
  # draw coastlines, state and country boundaries, edge of map.
  m.drawcoastlines(color='purple')
  m.drawstates()
  m.drawcountries()
  # draw parallels.
  parallels = np.arange(21,27,1.)
  m.drawparallels(parallels,labels=[1,1,0,0],fontsize=25)
  # draw meridians
  meridians = np.arange(119,124,1.)
  m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=25)
  lons, lats = np.meshgrid(lon,lat) 
  x, y = m(lons, lats) # compute map proj coordinates.
  # draw filled contours.
  cs_topo=m.contourf(x,y,topo,levels=levs,cmap='Greys',zorder=1)
  m.contour(x,y,topo,levels=levs[::3],colors='k',zorder=2)
  cs_sprec=m.contourf(x,y,masked_sprec,levels=levs_sprec,cmap=cm_sprec,zorder=3,alpha=0.9\
                      ,extend='max',interpolation='none')
  m.barbs(x[::step,::step],y[::step,::step],u[::step,::step],v[::step,::step],\
           length=6, pivot='middle',barbcolor='k',zorder=5)
  plt.title('Surface precipitation rate',fontsize=30)

  # color bar of topo
  cax_topo=fig.add_axes([0.1, 0.1, 0.7, 0.03])
  cbar_topo=plt.colorbar(cs_topo, cax=cax_topo, orientation='horizontal')
  cbar_topo.set_label('elevation(m)',fontsize=30)
    
        
  # color bar of lwp
  cax_sprec=fig.add_axes([0.87, 0.2, 0.03, 0.7])
  cbar_lwp=plt.colorbar(cs_sprec, cax=cax_sprec,ticks=levs_sprec[::15])
  cbar_lwp.set_label('precipitation rate(mm/hr)',fontsize=30)
  
  plt.suptitle('Local Time=%s [case:%s]'%(hhmm,case),fontsize=30)

  plt.savefig('%s/%s.%s.%s.png'%(pic_path,case,'sprec',hhmm))

  return

def setSprecColorbar(thd):
  #setting up sprec color bar
  levs_sprec=np.linspace(0.0,50.0,num=51)
  levs_sprec[0]=thd
  upper_color = plt.cm.jet(np.linspace(0.0, 50.0, num=51))[-1,:3]
  lower_color = plt.cm.jet(np.linspace(0.0, 50.0, num=51))[0,:3]
  cm_sprec=plt.cm.get_cmap('jet')
  cm_sprec.set_over(upper_color)
  cm_sprec.set_under(lower_color)
  return cm_sprec,levs_sprec


if __name__ == "__main__":
  case='tpe20050723cln' if not "-caseName" in sys.argv else sys.argv[sys.argv.index("-caseName")+1]
  dataDir=vvmDir+'/%s'%case

  #make sure case sub-directory in the postprocesses 
  path='./%s'%case
  if not os.path.isdir(path):
    os.mkdir(path)


  tStart=time.time()
  nt=getNT(dataDir)
  (nz,ny,nx),zc=getDimension('exp',dataDir)
  topo,utopo,vtopo,lon,lat,lon_edge,lat_edge=getTOPO(dataDir)

  print('caseName=%s'%case)
  print('Dimensions: nx=%d, ny=%d, nz=%d'%(nx,ny,nz))
  print('topo:',topo.min(),topo.max())
  print('lat:',lat.min(),lat.max())
  print('lon:',lon.min(),lon.max())

  # input tt, y,x
  #tt=81
  #xx=620
  #yy=710
  #rad=10.0
  #msk,ystart,yend,xstart,xend=getCircMsk(rad,xx,yy,nx,ny,dx=0.5)
  #b_sqare=calcBuoyancySqare(dataDir,'exp',tt,yy,xx,topo,zc,h_wd=20)
  #b_circle=calcBuoyancy(dataDir,'exp',xx,yy,tt,topo,zc,msk,ystart,yend,xstart,xend)
 
  #cape=calcCAPE(dataDir,'exp',81,topo,zc)
  #cape=calcCAPECutting(dataDir,'exp',81,topo,zc,620,680,690,720)
  #print(cape.shape,cape.max(),cape.min())

  #tEnd=time.time()
  #print('Calculating 1 timesteps using 1 processes...It costed %f sec'%(tEnd-tStart))

  
  #for tt in range(nt):
  for tt in [50,]:
    #PBLH=calcPBLHonTime(case,tt,topo,zc)
    #print('tt=%03d t:'%tt,PBLH.shape,PBLH.max(),PBLH.min())
    #np.save('%s/%s.PBLH.%03d.npy'%(path,case,tt),PBLH)
    u=getData(dataDir,'exp','u',tt)
    '''
    sprec=getData(dataDir,'exp','sprec',tt)
    np.save('%s/%s.sprec.%03d.npy'%(path,case,tt),sprec*3600)
    
    #th_sfc=getVarSfcOnTime(dataDir,case,'th',topo,tt,False)
    #print('tt=%03d th:'%tt,th_sfc.shape,th_sfc.max()-273.15,th_sfc.min()-273.15)
    #del th_sfc
    #t_sfc=getVarSfcOnTime(dataDir,case,'t',topo,tt,False)
    #print('tt=%03d t:'%tt,t_sfc.shape,t_sfc.max(),t_sfc.min())
    #np.save('%s/%s.tsfc.%03d.npy'%(path,case,tt),t_sfc)
    #del t_sfc
    u_sfc=getVarSfcOnTime(dataDir,'exp','u',topo,tt,True)
    print('tt=%03d u:'%tt,u_sfc.max(),u_sfc.min())
    np.save('%s/%s.usfc.%03d.npy'%(path,case,tt),u_sfc)
    #del u_sfc
    v_sfc=getVarSfcOnTime(dataDir,'exp','v',topo,tt,True)
    print('tt=%03d v:'%tt,v_sfc.max(),v_sfc.min())
    np.save('%s/%s.vsfc.%03d.npy'%(path,case,tt),v_sfc)
    #del v_sfc
    
    #lwp,CTH,CBH=calcCloudAndLWPonTime(case,tt,topo,zc)
    ##print(type(lwp),type(CTH),type(CBH))
    #print('tt=%03d lwp:'%tt,lwp.shape,lwp.max(),lwp.min())
    #print('tt=%03d CTH:'%tt,CTH.shape,np.nanmax(CTH),np.nanmin(CTH))
    #print('tt=%03d CBH:'%tt,CBH.shape,np.nanmax(CBH),np.nanmin(CBH))
    #np.save('%s/%s.lwp.%03d.npy'%(path,case,tt),lwp)
    #np.save('%s/%s.CTH.%03d.npy'%(path,case,tt),CTH)
    #np.save('%s/%s.CBH.%03d.npy'%(path,case,tt),CBH)
    #pAbd=calcPseAbd(lwp)
    #print('tt=%03d pAbd:'%tt,pAbd.shape,pAbd.shape,pAbd.max(),pAbd.min())
    #np.save('%s/%s.pAbd.%03d.npy'%(path,case,tt),pAbd)
    #del pAbd,lwp
    
    #LCL=getLCL(case,th,qv,zc)
    #print('tt=%03d LCL:'%tt,LCL.shape,np.nanmean(LCL),np.nanmax(LCL),np.nanmin(LCL))
    #np.save('%s.LCL.%03d.npy'%(case,tt),LCL)
    '''
