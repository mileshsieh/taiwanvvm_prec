import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import vvmDataUtil as vvm
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def get_truncated_gist_yarg_from_10(name="MPL_gist_yarg_10"):
    """
    Create a Matplotlib colormap equivalent to NCL-style MPL_gist_yarg[10:].

    This removes the first 10 colors from the 256-color gist_yarg table.
    """
    base_cmap = plt.get_cmap("gist_yarg", 256)
    colors = base_cmap(np.arange(256))

    # Equivalent to NCL MPL_gist_yarg[10:]
    colors = colors[10:]

    new_cmap = LinearSegmentedColormap.from_list(
        name,
        colors,
        N=len(colors),
    )

    return new_cmap

def lon_to_xkm(lon, section_lat):
    """
    Convert longitude coordinate to along-section distance in km
    for a constant-latitude section.

    Parameters
    ----------
    lon : 1D array
        Longitude coordinate.

    section_lat : float
        Section latitude in degrees north.

    Returns
    -------
    xkm : 1D array
        Along-section distance in km, with xkm[0] = 0.
    """
    lon = np.asarray(lon)
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(section_lat))
    xkm = (lon - lon[0]) * km_per_deg_lon
    return xkm

def add_section_inset_map(
    ax_parent,
    map_lon,
    map_lat,
    map_topo,
    section_lon,
    section_lat,
    *,
    inset_box=(0.03, 0.68, 0.23, 0.23),
    map_extent=None,
    topo_unit="m",
    cmap="terrain",
    section_color="red",
    coast_color="k",
    title="Section",
):
    """
    Add an inset map showing Taiwan topography and the section location.

    Parameters
    ----------
    ax_parent : matplotlib axes
        Parent axes, usually the upper X-Z panel ax_xz.

    map_lon : 1D array, shape (nx,)
        Longitude of the 2D topography map.

    map_lat : 1D array, shape (ny,)
        Latitude of the 2D topography map.

    map_topo : 2D array, shape (ny, nx)
        Topography over Taiwan/domain.

    section_lon : 1D array
        Longitudes of the selected cross section.

    section_lat : float
        Latitude of the cross section.

    inset_box : tuple
        (x0, y0, width, height) in parent-axes coordinates.

    map_extent : tuple or None
        (lonmin, lonmax, latmin, latmax).
        If None, use full domain extent.

    topo_unit : {"m", "km"}
        Unit of map_topo.

    cmap : str or Colormap
        Colormap for terrain.

    section_color : str
        Color of section line and points.

    coast_color : str
        Color of coastline/topography-zero contour.

    title : str
        Small inset title.
    """

    map_lon = np.asarray(map_lon)
    map_lat = np.asarray(map_lat)
    map_topo = np.asarray(map_topo)
    section_lon = np.asarray(section_lon)

    if map_topo.shape != (map_lat.size, map_lon.size):
        raise ValueError(
            f"map_topo must have shape ({map_lat.size}, {map_lon.size}), "
            f"but got {map_topo.shape}"
        )

    # Convert topo to meters for map display if needed
    topo_plot = map_topo.copy()
    if topo_unit == "km":
        topo_plot = topo_plot * 1000.0
    elif topo_unit != "m":
        raise ValueError("topo_unit must be 'm' or 'km'")

    Lon2d, Lat2d = np.meshgrid(map_lon, map_lat)

    # Create inset axes inside parent axes
    ax_inset = ax_parent.inset_axes(inset_box)

    # Terrain shading
    topo_masked = np.ma.masked_where(topo_plot <= 0, topo_plot)
    pcm = ax_inset.pcolormesh(
        Lon2d,
        Lat2d,
        topo_masked,
        cmap=cmap,
        shading="auto",
        vmin=0,
        vmax=np.nanmax(topo_plot) if np.nanmax(topo_plot) > 0 else 3000,
        zorder=11,
    )

    # Coastline-like contour from topo = 0
    ax_inset.contour(
        Lon2d,
        Lat2d,
        topo_plot,
        levels=[0.0],
        colors=coast_color,
        linewidths=0.8,
        zorder=12,
    )

    # Section line
    ax_inset.plot(
        [section_lon[0], section_lon[-1]],
        [section_lat, section_lat],
        color=section_color,
        linewidth=2.0,
        zorder=13,
    )

    # Endpoints A and B
    ax_inset.plot(section_lon[0], section_lat, "o", color=section_color, ms=4, zorder=14)
    ax_inset.plot(section_lon[-1], section_lat, "o", color=section_color, ms=4, zorder=14)

    ax_inset.text(
        section_lon[0],
        section_lat,
        "A",
        color=section_color,
        fontsize=8,
        ha="right",
        va="bottom",
        zorder=15,
    )
    ax_inset.text(
        section_lon[-1],
        section_lat,
        "B",
        color=section_color,
        fontsize=8,
        ha="left",
        va="bottom",
        zorder=15,
    )

    # Map extent
    if map_extent is None:
        ax_inset.set_xlim(map_lon.min(), map_lon.max())
        ax_inset.set_ylim(map_lat.min(), map_lat.max())
    else:
        ax_inset.set_xlim(map_extent[0], map_extent[1])
        ax_inset.set_ylim(map_extent[2], map_extent[3])

    # Formatting
    ax_inset.set_title(title, fontsize=8, pad=2)
    ax_inset.set_xlabel("")
    ax_inset.set_ylabel("")
    ax_inset.tick_params(labelsize=7, length=2)

    for spine in ax_inset.spines.values():
        spine.set_linewidth(0.8)
    ax_inset.tick_params(axis="x", labelbottom=False)
    ax_inset.tick_params(axis="y", labelleft=False)
    return ax_inset

def plot_XZ(lon,zc,topo,precip,u,w,qi,qc,qr,section_lat,title,lon_label_left="A",lon_label_right="B",zlim=(0, 18)):
    lon = np.asarray(lon)
    zc = np.asarray(zc)
    topo = np.asarray(topo)
    precip = np.asarray(precip)

    u = np.asarray(u)
    w = np.asarray(w)
    qi = np.asarray(qi)
    qc = np.asarray(qc)
    qr = np.asarray(qr)

    nz, nx = u.shape
    xkm = lon_to_xkm(lon, section_lat)
    X2d, Z2d = np.meshgrid(xkm, zc/1000.0)

    q_cloud = (qi + qc) * 1000.0
    q_rain = qr * 1000.0
    qi_qc_levels = np.linspace(0.0, 5.5, 11)

    figsize=(11,8)
    dpi=100
    plt.close()
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(2,1,height_ratios=[4.5, 1.5],hspace=0.0,figure=fig,)
    ax_xz = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[1, 0], sharex=ax_xz)
    #q_cloud
    cloud_cmap = get_truncated_gist_yarg_from_10()
    cf = ax_xz.contourf(X2d,Z2d,q_cloud,levels=qi_qc_levels,
        cmap=cloud_cmap,vmin=qi_qc_levels.min(),vmax=qi_qc_levels.max(),extend="max")
    #colorbar
    cax = ax_xz.inset_axes([1.025, 0.08, 0.025, 0.85],  # [x0, y0, width, height] in ax_xz coordinates
        transform=ax_xz.transAxes,)
    cbar = fig.colorbar(cf, cax=cax)
    cbar.set_label(r"$q_i + q_c$ (g/kg)")
    #q_rain
    ax_xz.contour(X2d,Z2d,q_rain,levels=[0.5,],colors="dodgerblue",linewidths=2.0)
    #wind
    w_scale_factor=1
    quiver_scale=5
    u_pos_color="seagreen"
    u_neg_color="firebrick"
    xs = slice(None, None, 4) #skip x
    zs = slice(None, None, 2) #skip z

    x_q = X2d[zs, xs]
    z_q = Z2d[zs, xs]
    u_q = u[zs, xs]
    w_q = w[zs, xs] * w_scale_factor

    mask_pos = u_q >= 0.5
    mask_neg = u_q <= 0.5

    q_pos = ax_xz.quiver(x_q[mask_pos],z_q[mask_pos],u_q[mask_pos],w_q[mask_pos],
        color=u_pos_color,angles="xy",scale_units="xy",
        scale=quiver_scale,alpha=0.9,zorder=6,)
    ax_xz.quiver(x_q[mask_neg],z_q[mask_neg],u_q[mask_neg],w_q[mask_neg],
        color=u_neg_color,angles="xy",scale_units="xy",
        scale=quiver_scale,alpha=0.9,zorder=6,)

    ax_xz.quiverkey(q_pos,X=1.045,Y=0.98,U=10.0,label="10 m/s",
        labelpos="E",coordinates="axes",fontproperties={"size": 10})

    #topo
    ax_xz.fill_between(xkm,0.0,topo,step="post",color="black",zorder=10,)
    ax_xz.step(xkm,topo,where="post",color="black",linewidth=1.2,zorder=11,)

    xlim = (np.nanmin(xkm), np.nanmax(xkm))
    ax_xz.set_xlim(xlim)
    ax_xz.set_ylim(0.0, 18.0)
    ax_xz.set_ylabel("z (km)")
    if title is not None:
        ax_xz.set_title(title, fontsize=18, pad=8)
    # Keep x tick locations for alignment, but do not show labels
    tick_locs = np.linspace(xkm[0], xkm[-1], 5)
    ax_xz.set_xticks(tick_locs)
    ax_xz.tick_params(axis="x", labelbottom=False)
    #preci
    ax_p.plot(xkm,precip,color="tab:blue",linewidth=2.2,label="Precip. Rate",)
    ax_p.set_xlim(xlim)
    ax_p.set_ylabel(f"(mm/hr)")
    ax_p.set_xlabel("")
    ax_p.grid(True, alpha=0.25)
    ax_p.legend(loc="upper left", frameon=True)

    #ymax = np.nanmax(precip)
    #if np.isfinite(ymax) and ymax > 0:
    #    ax_p.set_ylim(0.0, ymax * 1.15)
    #else:
    #    ax_p.set_ylim(0.0, 1.0)
    ax_p.set_ylim(0.0, 90.0)

    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(section_lat))
    if False:
        ax_p.set_xticks([xkm[0], xkm[-1]])
        ax_p.set_xticklabels(
            [
                f"{lon_label_left}\n({lon[0]:.1f}E, {section_lat:.1f}N)",
                f"{lon_label_right}\n({lon[-1]:.1f}E, {section_lat:.1f}N)",
            ]
        )
    else:
        tick_lons = lon[0] + tick_locs / km_per_deg_lon

        ax_p.set_xticks(tick_locs)
        ax_p.set_xticklabels([f"{lo:.2f}E" for lo in tick_lons])
    fig.align_ylabels([ax_xz, ax_p])

    # Reserve right margin for the external quiver key and colorbar.
    # Increase right if labels are clipped.
    fig.subplots_adjust(left=0.10,right=0.86,bottom=0.10,top=0.92,hspace=0.0)

    return fig, (ax_xz, ax_p)

def sample_taiwanvvm_xz_section_by_slice(
    lon,
    lat,
    zc,
    u,
    w,
    qi,
    qc,
    qr,
    topo,
    *,
    lon_start,
    lon_end,
    section_lat,
    precip=None,
    topo_unit="m",
    output_topo_unit="km",
    verbose=True,
):
    """
    Slice a constant-latitude X-Z section directly from TaiwanVVM outputs
    without interpolation.

    This function finds the nearest model-grid latitude to `section_lat`,
    finds the nearest longitude indices to `lon_start` and `lon_end`,
    and directly slices the original arrays.

    Parameters
    ----------
    lon : ndarray, shape (nx,)
        Longitude coordinate.

    lat : ndarray, shape (ny,)
        Latitude coordinate.

    zc : ndarray, shape (nz,)
        Vertical coordinate. Usually in km for plotting.

    u, w, qi, qc, qr : ndarray, shape (nz, ny, nx)
        TaiwanVVM 3D output fields.

    topo : ndarray, shape (ny, nx)
        Terrain height.

    lon_start, lon_end : float
        Starting and ending longitude of the section.

    section_lat : float
        Target latitude of the section. The nearest model-grid latitude
        will be used.

    precip : ndarray, shape (ny, nx), optional
        Surface precipitation rate. If provided, it will be sliced along
        the same section.

    topo_unit : {"m", "km"}
        Unit of input topography.

    output_topo_unit : {"m", "km"}
        Unit of returned topography.

    verbose : bool
        If True, print selected indices and actual coordinates.

    Returns
    -------
    section : dict
        Dictionary containing sliced section data.

        section["lon"]        : shape (nsec,)
        section["lat"]        : scalar, actual sliced latitude
        section["zc"]         : shape (nz,)
        section["u"]          : shape (nz, nsec)
        section["w"]          : shape (nz, nsec)
        section["qi"]         : shape (nz, nsec)
        section["qc"]         : shape (nz, nsec)
        section["qr"]         : shape (nz, nsec)
        section["topo"]       : shape (nsec,)
        section["precip"]     : shape (nsec,) or None
        section["j_sec"]      : selected latitude index
        section["i_start"]    : selected longitude start index
        section["i_end"]      : selected longitude end index
    """

    lon = np.asarray(lon)
    lat = np.asarray(lat)
    zc = np.asarray(zc)

    u = np.asarray(u)
    w = np.asarray(w)
    qi = np.asarray(qi)
    qc = np.asarray(qc)
    qr = np.asarray(qr)
    topo = np.asarray(topo)

    nz, ny, nx = u.shape

    # ------------------------------------------------------------
    # Basic shape checks
    # ------------------------------------------------------------
    if lon.shape != (nx,):
        raise ValueError(f"lon must have shape ({nx},), but got {lon.shape}")

    if lat.shape != (ny,):
        raise ValueError(f"lat must have shape ({ny},), but got {lat.shape}")

    if zc.shape != (nz,):
        raise ValueError(f"zc must have shape ({nz},), but got {zc.shape}")

    for name, arr in {
        "w": w,
        "qi": qi,
        "qc": qc,
        "qr": qr,
    }.items():
        if arr.shape != (nz, ny, nx):
            raise ValueError(
                f"{name} must have shape {(nz, ny, nx)}, but got {arr.shape}"
            )

    if topo.shape != (ny, nx):
        raise ValueError(f"topo must have shape {(ny, nx)}, but got {topo.shape}")

    if precip is not None:
        precip = np.asarray(precip)
        if precip.shape != (ny, nx):
            raise ValueError(
                f"precip must have shape {(ny, nx)}, but got {precip.shape}"
            )

    # ------------------------------------------------------------
    # Find nearest grid indices
    # ------------------------------------------------------------
    j_sec = int(np.argmin(np.abs(lat - section_lat)))

    i0 = int(np.argmin(np.abs(lon - lon_start)))
    i1 = int(np.argmin(np.abs(lon - lon_end)))

    # Allow section from west to east or east to west.
    if i0 <= i1:
        i_slice = slice(i0, i1 + 1)
        step_sign = 1
    else:
        i_slice = slice(i0, i1 - 1, -1)
        step_sign = -1

    lon_sec = lon[i_slice]
    lat_sec = lat[j_sec]

    # ------------------------------------------------------------
    # Direct slicing: no interpolation
    # ------------------------------------------------------------
    u_xz = u[:, j_sec, i_slice]
    w_xz = w[:, j_sec, i_slice]
    qi_xz = qi[:, j_sec, i_slice]
    qc_xz = qc[:, j_sec, i_slice]
    qr_xz = qr[:, j_sec, i_slice]

    topo_xz = topo[j_sec, i_slice]

    if precip is not None:
        precip_xz = precip[j_sec, i_slice]
    else:
        precip_xz = None

    # ------------------------------------------------------------
    # Topography unit conversion
    # ------------------------------------------------------------
    if topo_unit == output_topo_unit:
        pass
    elif topo_unit == "m" and output_topo_unit == "km":
        topo_xz = topo_xz / 1000.0
    elif topo_unit == "km" and output_topo_unit == "m":
        topo_xz = topo_xz * 1000.0
    else:
        raise ValueError("topo_unit and output_topo_unit must be 'm' or 'km'")

    if verbose:
        print("Selected X-Z section:")
        print(f"  requested section_lat = {section_lat}")
        print(f"  selected j_sec        = {j_sec}")
        print(f"  actual lat[j_sec]     = {lat_sec}")
        print("")
        print(f"  requested lon_start   = {lon_start}")
        print(f"  selected i_start      = {i0}")
        print(f"  actual lon[i_start]   = {lon[i0]}")
        print("")
        print(f"  requested lon_end     = {lon_end}")
        print(f"  selected i_end        = {i1}")
        print(f"  actual lon[i_end]     = {lon[i1]}")
        print("")
        print(f"  section direction     = {'west-to-east' if step_sign == 1 else 'east-to-west'}")
        print(f"  number of x points    = {lon_sec.size}")

    section = {
        "lon": lon_sec,
        "lat": lat_sec,
        "zc": zc,
        "u": u_xz,
        "w": w_xz,
        "qi": qi_xz,
        "qc": qc_xz,
        "qr": qr_xz,
        "topo": topo_xz,
        "precip": precip_xz,
        "j_sec": j_sec,
        "i_start": i0,
        "i_end": i1,
        "lon_start_actual": lon[i0],
        "lon_end_actual": lon[i1],
        "section_lat_requested": section_lat,
        "section_lat_actual": lat_sec,
    }

    return section

def lon_to_xkm(lon, section_lat):
    """
    Convert longitude coordinate to distance in km along a constant-latitude section.
    The first longitude point is defined as x = 0 km.
    """
    lon = np.asarray(lon)
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(section_lat))
    xkm = (lon - lon[0]) * km_per_deg_lon
    return xkm



if __name__=='__main__':
    
    caseList=[a.split('/')[-1] for a in glob.glob('/data2/VVM/taiwanvvm_lc/taiwanvvm/lc*')]
    print(len(caseList))
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

    for tt in range(48,121):
    #for tt in [93,]:
        a=stime+np.timedelta64(dt*tt,'m')
        hhmm=pd.to_datetime(str(a)).strftime('%H:%M')
        print(hhmm)

        u_intp,v_intp,w_intp=vvm.get3DWind_intp(dataDir,case,tp,utp,vtp,tt)
        sprec=vvm.getData(dataDir,case,'sprec',tt)
        qc=vvm.getData(dataDir,case,'qc',tt)
        qr=vvm.getData(dataDir,case,'qr',tt)
        qi=vvm.getData(dataDir,case,'qi',tt)
        section = sample_taiwanvvm_xz_section_by_slice(
            lon=lon,
            lat=lat,
            zc=zc,
            u=u_intp,
            w=w_intp,
            qi=qi,
            qc=qc,
            qr=qr,
            topo=topo,
            precip=sprec,
            lon_start=120.5,
            lon_end=121.5,
            section_lat=24.5,
            topo_unit="m",
            output_topo_unit="km",
        )
        fig, (ax_xz, ax_p)=plot_XZ(section["lon"],section["zc"],section["topo"],section["precip"],section["u"],section["w"],
                                   section["qi"],section["qc"],section["qr"],24.5,f'{ymd} {hhmm} LST')

        inset_box=(0.03, 0.58, 0.12, 0.35)
        map_extent=(119.8, 122.3, 21.7, 25.5)
        add_section_inset_map(ax_parent=ax_xz,map_lon=lon,map_lat=lat,map_topo=topo,
            section_lon=np.asarray(section["lon"]),section_lat=24.5,
            inset_box=inset_box,map_extent=map_extent,
            topo_unit="m",cmap="terrain",section_color="red",coast_color="k",title="",)

        fig.savefig(f'../figures/xz_cloud.{case}.t{tt:03}.png', bbox_inches="tight", dpi=100)
