import numpy as np
from matplotlib import pyplot as plt
xyLL={
        'sw':[262,320],
        'north':[481,673],
        'south':[323,120],
        }
lon=np.load('../data/lon.npy')
lat=np.load('../data/lat.npy')
topo=np.load('../data/topo.npy')
plt.close()
plt.figure(figsize=(8,8))
plt.contourf(lon,lat,np.where(topo>0,topo,np.nan),cmap='Greens')
plt.contour(lon,lat,topo,levels=[0.5,],colors=['k',],zorder=6)
for a in xyLL.keys():
    print(a,lon[xyLL[a][0]],lon[xyLL[a][0]+256],lat[xyLL[a][1]],lat[xyLL[a][1]+256])
    xs,xe,ys,ye=lon[xyLL[a][0]],lon[xyLL[a][0]+256],lat[xyLL[a][1]],lat[xyLL[a][1]+256]
    plt.plot([xs,xe,xe,xs,xs],[ys,ys,ye,ye,ys],lw=4)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.xticks([])
plt.yticks([])
plt.savefig('../figures/domains.png')
