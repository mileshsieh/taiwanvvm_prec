import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import vvmDataUtil as vvm
import matplotlib
homeDir='/data2/VVM/taiwanvvm_lc'
case='lc_20220625'

def draw_utopo_boundary_on_theta_grid(ax, utopo, dx=1.0, dz=1.0):
    """
    Draw terrain-mask boundary for interpolated u on theta/scalar grid.

    Parameters
    ----------
    ax : matplotlib axis

    utopo : ndarray, shape (nx,)
        Terrain-mask height for interpolated u values on theta grid.

        If utopo[i] = m:
            k = 0, ..., m-1 are masked / terrain-affected
            k = m is the first valid theta-grid level for interpolated u.

    dx, dz : float
        Grid spacing.

    Notes
    -----
    This draws a block-like green dashed boundary over scalar grid columns.
    Each column i spans x = i*dx to x = (i+1)*dx.
    """

    utopo = np.asarray(utopo).astype(int)
    nx = utopo.size

    for i in range(nx):
        x0 = i * dx
        x1 = (i + 1) * dx
        z = utopo[i] * dz

        # horizontal top boundary of the mask in this scalar column
        ax.plot(
            [x0, x1],
            [z, z],
            color="green",
            linestyle="--",
            linewidth=2.0,
            zorder=10,
        )

        # vertical jump at the left edge if needed
        if i > 0:
            z_prev = utopo[i - 1] * dz
            if z_prev != z:
                ax.plot(
                    [x0, x0],
                    [z_prev, z],
                    color="green",
                    linestyle="--",
                    linewidth=2.0,
                    zorder=10,
                )

def plot_xz_cgrid_check(
    topo,
    u=None,
    w=None,
    theta=None,
    utopo=None,
    dx=1.0,
    dz=1.0,
    u_fmt="{:.2f}",
    w_fmt="{:.2f}",
    show_u_values=True,
    show_w_values=False,
    show_theta_values=False,
    show_values_inside_terrain=False,
    show_utopo_boundary=False,
    title="XZ C-grid terrain mask and interpolation check",
    figsize=(12, 8),
):
    """
    Plot an XZ schematic for Arakawa C-grid data with block-mountain topography.

    Parameters
    ----------
    topo : ndarray, shape (nx,)
        Terrain height in integer model layers.
        topo[i] = number of solid cells below the fluid column at x-index i.

    u : ndarray, optional, shape (nz, nx+1)
        Zonal wind on x-staggered faces.
        u[k, i_face] is located at x = i_face * dx,
        z = (k + 0.5) * dz.

    w : ndarray, optional, shape (nz+1, nx)
        Vertical wind on z-staggered faces.
        w[k_face, i] is located at x = (i + 0.5) * dx,
        z = k_face * dz.

    theta : ndarray, optional, shape (nz, nx)
        Scalar field on cell centers.
        theta[k, i] is located at x = (i + 0.5) * dx,
        z = (k + 0.5) * dz.

    dx, dz : float
        Grid spacing in x and z.

    show_u_values, show_w_values, show_theta_values : bool
        Whether to print values on the corresponding grid points.

    mask_u_below_topo, mask_w_below_topo : bool
        Whether to visually suppress values below terrain.

    Notes
    -----
    This plotting function assumes a block-mountain representation:
    each scalar cell (i, k) is solid if k < topo[i].
    """

    topo = np.asarray(topo).astype(int)
    nx = topo.size

    # Infer nz
    nz_candidates = []

    if theta is not None:
        theta = np.asarray(theta)
        nz_candidates.append(theta.shape[0])
        assert theta.shape[1] == nx, (
            f"theta should have shape (nz, nx), but got {theta.shape}"
        )

    if u is not None:
        u = np.asarray(u)
        nz_candidates.append(u.shape[0])
        assert u.shape[1] == nx + 1, (
            f"u should have shape (nz, nx+1), but got {u.shape}"
        )

    if w is not None:
        w = np.asarray(w)
        nz_candidates.append(w.shape[0] - 1)
        assert w.shape[1] == nx, (
            f"w should have shape (nz+1, nx), but got {w.shape}"
        )
    if utopo is not None:
        utopo = np.asarray(utopo).astype(int)

        if utopo.shape != (nx,):
            raise ValueError(
                f"utopo should have shape (nx,), because it is defined on theta/scalar columns. "
                f"Got utopo.shape={utopo.shape}, while nx={nx}."
            )


    if len(nz_candidates) == 0:
        nz = int(np.nanmax(topo)) + 3
    else:
        nz = max(nz_candidates)

    assert np.nanmax(topo) <= nz, "topo contains values higher than nz."

    fig, ax = plt.subplots(figsize=figsize)

    # ------------------------------------------------------------
    # 1. Draw terrain blocks
    # ------------------------------------------------------------
    for i in range(nx):
        for k in range(topo[i]):
            rect = Rectangle(
                (i * dx, k * dz),
                dx,
                dz,
                facecolor="0.6",
                edgecolor="k",
                linewidth=0.8,
                alpha=0.85,
            )
            ax.add_patch(rect)

    # ------------------------------------------------------------
    # 2. Draw scalar-cell grid lines
    # ------------------------------------------------------------
    for i in range(nx + 1):
        ax.axvline(i * dx, color="k", linewidth=0.8)

    for k in range(nz + 1):
        ax.axhline(k * dz, color="k", linewidth=0.8)

    # ------------------------------------------------------------
    # 3. Draw staggered half-level reference lines
    # ------------------------------------------------------------
    for i in range(nx):
        ax.axvline((i + 0.5) * dx, color="k", linestyle="--", linewidth=0.8, alpha=0.5)

    for k in range(nz):
        ax.axhline((k + 0.5) * dz, color="k", linestyle="--", linewidth=0.8, alpha=0.5)

    if utopo is not None and show_utopo_boundary:
        draw_utopo_boundary_on_theta_grid(ax, utopo, dx=dx, dz=dz)
    # ------------------------------------------------------------
    # 4. Helper functions for terrain masking
    # ------------------------------------------------------------
    def is_scalar_inside_terrain(k, i):
        """
        Scalar/theta point is at cell center.
    
        topo[i] = number of solid scalar cells in column i.
    
        For scalar cell center:
        - k < topo[i]  : inside terrain
        - k >= topo[i] : atmosphere
        """
        return k < topo[i]
    
    
    def is_w_inside_terrain(k_face, i):
        """
        w point is located at vertical cell interface.
    
        For block mountain:
        - k_face < topo[i]  : inside terrain
        - k_face == topo[i] : terrain surface, always shown
        - k_face > topo[i]  : atmosphere, always shown
        """
        return k_face < topo[i]
    
    
    def is_u_inside_terrain(k, i_face):
        """
        u point is located at x-face and vertical cell center.
    
        It lies between column i_face-1 and i_face.
    
        We define true terrain interior only when BOTH adjacent columns
        contain terrain at level k.
    
        Therefore:
        - both adjacent columns are terrain : inside terrain
        - one terrain and one atmosphere    : terrain side boundary, always shown
        - both atmosphere                   : atmosphere, always shown
    
        Boundary faces use the nearest physical column.
        """
    
        if i_face == 0:
            return k < topo[0]
    
        elif i_face == nx:
            return k < topo[nx - 1]
    
        else:
            left_solid = k < topo[i_face - 1]
            right_solid = k < topo[i_face]
    
            return left_solid and right_solid
    # ------------------------------------------------------------
    # 5. Plot theta locations and optionally values
    # ------------------------------------------------------------
    if theta is not None:
        for k in range(theta.shape[0]):
            for i in range(nx):
                x = (i + 0.5) * dx
                z = (k + 0.5) * dz

                inside_terrain = is_scalar_inside_terrain(k, i)

                if inside_terrain and not show_values_inside_terrain:
                    continue

                #ax.plot(x, z, "o", color="tab:blue", markersize=4)

                if show_theta_values:
                    ax.text(
                        x,
                        z,
                        f"{theta[k, i]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="tab:blue",
                    )

    # ------------------------------------------------------------
    # 6. Plot u locations and values
    # ------------------------------------------------------------
    if u is not None:
        for k in range(u.shape[0]):
            for i_face in range(u.shape[1]):
                x = i_face * dx
                z = (k + 0.5) * dz

                inside_terrain = is_u_inside_terrain(k, i_face)

                if inside_terrain and not show_values_inside_terrain:
                    continue

                #ax.plot(x, z, "o", color="red", markersize=4)

                if show_u_values:
                    ax.text(
                        x,
                        z,
                        u_fmt.format(u[k, i_face]),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="red",
                        bbox=dict(
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.65,
                            pad=0.5,
                        ),
                    )

    # ------------------------------------------------------------
    # 7. Plot w locations and optionally values
    # ------------------------------------------------------------
    if w is not None:
        for k_face in range(w.shape[0]):
            for i in range(nx):
                x = (i + 0.5) * dx
                z = k_face * dz

                inside_terrain = is_w_inside_terrain(k_face, i)

                if inside_terrain and not show_values_inside_terrain:
                    continue

                #ax.plot(x, z, "^", color="black", markersize=4)

                if show_w_values:
                    ax.text(
                        x,
                        z,
                        w_fmt.format(w[k_face, i]),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="black",
                        bbox=dict(
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.65,
                            pad=0.5,
                        ),
                    )

    # ------------------------------------------------------------
    # 8. Annotate topo values
    # ------------------------------------------------------------
    for i in range(nx):
        ax.text(
            (i + 0.5) * dx,
            -0.45 * dz,
            f"TOPO={topo[i]}",
            ha="center",
            va="top",
            fontsize=11,
        )

    # ------------------------------------------------------------
    # 9. Axis settings
    # ------------------------------------------------------------
    ax.set_xlim(-0.2 * dx, (nx + 0.2) * dx)
    ax.set_ylim(-0.8 * dz, (nz + 0.3) * dz)

    ax.set_xlabel("x index")
    ax.set_ylabel("z / k index")
    ax.set_title(title)

    ax.set_aspect("equal", adjustable="box")

    ax.set_xticks(np.arange(nx + 1) * dx)
    ax.set_yticks(np.arange(nz + 1) * dz)

    ax.grid(False)

    plt.tight_layout()
    return fig, ax

if __name__=='__main__':
    # case: 20220625
    homeDir='/data2/VVM/taiwanvvm_lc'
    case='lc_20220625'

    #get data
    dataDir=homeDir+'/%s'%case
    nt=vvm.getNT(dataDir)
    tp,utp,vtp,lon,lat,lon_edge,lat_edge=vvm.getTOPO(dataDir)
    tt=72
    u3D=vvm.getData(dataDir,case,'u',tt)
    v3D=vvm.getData(dataDir,case,'v',tt)
    w3D=vvm.getData(dataDir,case,'w',tt)
    qv3D=vvm.getData(dataDir,case,'qv',tt)*1000.0
    uintp, vintp = vvm.get3DWind_intp(dataDir,case,tp,utp,vtp,tt)

    nx = 15
    nz = 10

    xx=621
    '''
    #for XZ check
    yy=842
    topo = tp[yy,xx:xx+nx]
    utopo = utp[yy,xx:xx+nx]
    u = u3D[:nz,yy,xx-1:xx+nx] 
    w = w3D[:nz+1,yy,xx:xx+nx]
    #theta = qv3D[:nz,yy,xx:xx+nx]
    theta = uintp[:nz,yy,xx:xx+nx]
    '''
    #for YZ check
    
    yy=630
    k_offset=6
    topo = tp[yy:yy+nx,xx]-k_offset
    utopo = utp[yy:yy+nx,xx]-k_offset
    u = v3D[k_offset:k_offset+nz,yy-1:yy+nx,xx] 
    w = w3D[k_offset:k_offset+nz+1,yy:yy+nx,xx]
    #theta = qv3D[:nz,yy,xx:xx+nx]
    theta = vintp[k_offset:k_offset+nz,yy:yy+nx,xx]
    
    plt.close() 
    fig, ax = plot_xz_cgrid_check(
        topo=topo,
        u=u,
        w=w,
        theta=theta,
        #utopo=utopo,
        show_u_values=True,
        show_w_values=True,
        show_theta_values=True,
        show_values_inside_terrain=True,
        #show_utopo_boundary=False,
        #title=f"C-grid XZ terrain-mask check(x={xx}:{xx+nx},y={yy})",
        title=f"C-grid YZ terrain-mask check (x={xx},y={yy}:{yy+nx},k_offset={k_offset})",
        figsize=(20, 10)
    )
    
