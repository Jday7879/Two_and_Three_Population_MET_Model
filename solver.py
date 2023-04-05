# -*- coding: utf-8 -*-
"""
Module will be imported and run until a steady state is found for the fluid system

This will take boundary and resolution as inputs and return ux uy and psi for the system

"""

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg
import time


def steady_state(boundary, psi, nx, ny, dx, dy, error=1e-8, dt_min=1e2, storage_interval=180):
    start_time = time.time()
    nxy = nx * ny
    nxy_one = (nx + 1) * (ny + 1)

    nu = 1e-6  # fluid kinematic viscosity
    dt_max_diffusion = (dx ** 2 * dy ** 2) / (2 * nu * (dx ** 2 + dy ** 2))
    vorticity = np.zeros((nx, ny))  # This is vorticity
    rhs = np.zeros((nx + 1, ny + 1))  # This will be the Laplacian of psi
    fx = np.zeros((nx + 1, ny))  # Rightward vorticity flux
    fy = np.zeros((nx, ny + 1))  # Upward vorticity flux
    d_vorticity_dt1 = np.zeros((nx, ny))  # initalised for rk4 algorithm
    d_vorticity_dt2 = np.zeros((nx, ny))
    diagonal_x = np.array([[1 / (dx ** 2)], [-2 / (dx ** 2)], [1 / (dx ** 2)]]).repeat(nx + 1, axis=1)
    diagonal_y = np.array([[1 / (dy ** 2)], [-2 / (dy ** 2)], [1 / (dy ** 2)]]).repeat(ny + 1, axis=1)
    positions = [-1, 0, 1]

    Ax = sp.sparse.spdiags(diagonal_x, positions, nx + 1, nx + 1)
    Ay = sp.sparse.spdiags(diagonal_y, positions, ny + 1, ny + 1)
    Iy = sp.sparse.eye(ny + 1)
    Ix = sp.sparse.eye(nx + 1)
    AA = sp.sparse.kron(Iy, Ax) + sp.sparse.kron(Ay, Ix)
    bd = boundary.T
    bd = bd.reshape((nxy_one))
    AA = AA.tolil()  # converting to alternate form of sparse matrix
    st_loop = time.time()
    for i in np.arange(nxy_one):
        if bd[i] == 1:
            AA[i, :] = 0
            AA[i, i] = 1
    AA = AA.tocsr()
    LU = sp.sparse.linalg.factorized(AA.tocsc())
    print('AA matrix defined', time.time() - st_loop)

    diagonal_x = np.array([[1 / (dx ** 2)], [-2 / (dx ** 2)], [1 / (dx ** 2)]]).repeat(nx, axis=1)
    diagonal_y = np.array([[1 / (dy ** 2)], [-2 / (dy ** 2)], [1 / (dy ** 2)]]).repeat(ny, axis=1)
    Bx = sp.sparse.spdiags(diagonal_x, positions, nx, nx)
    By = sp.sparse.spdiags(diagonal_y, positions, ny, ny)
    Iy = sp.sparse.eye(ny)
    Ix = sp.sparse.eye(nx)
    BB = sp.sparse.kron(Iy, Bx) + sp.sparse.kron(By, Ix)
    CC = sp.sparse.csr_matrix((nxy, nxy_one))
    CC = CC.tolil()  # Faster to use this form for indexing vs csr format
    BB = BB.tolil()

    for ii in np.arange(nxy):
        ix = int((ii) % nx)
        iy = int(np.floor((ii) / nx))
        jj = (iy) * (nx + 1) + ix
        if boundary[ix, iy] * boundary[ix, iy + 1] == 1:  # Boundary on left
            BB[ii, ii] = BB[ii, ii] - 2 / (dx ** 2)
            BB[ii, ii + 1] = BB[ii, ii + 1] * 4 / 3
            if ix != 1:
                BB[ii, ii - 1] = 0
            CC[ii, jj] = CC[ii, jj] + 8 / 3 / (dx ** 4)
            CC[ii, jj + nx + 1] = CC[ii, jj + nx + 1] + 8 / 3 / (dx ** 4)
            CC[ii, jj + 1] = CC[ii, jj + 1] - 8 / 3 / (dx ** 4)
            CC[ii, jj + nx + 2] = CC[ii, jj + nx + 2] - 8 / 3 / (dx ** 4)

        if boundary[ix + 1, iy] * boundary[ix + 1, iy + 1] == 1:  # Boundary on right
            BB[ii, ii] = BB[ii, ii] - 2 / (dx ** 2)
            BB[ii, ii - 1] = BB[ii, ii - 1] * 4 / 3
            if ix != nx - 1:
                BB[ii, ii + 1] = 0
            CC[ii, jj] = CC[ii, jj] - 8 / 3 / (dx ** 4)
            CC[ii, jj + nx + 1] = CC[ii, jj + nx + 1] - 8 / 3 / (dx ** 4)
            CC[ii, jj + 1] = CC[ii, jj + 1] + 8 / 3 / (dx ** 4)
            CC[ii, jj + nx + 2] = CC[ii, jj + nx + 2] + 8 / 3 / (dx ** 4)

        if boundary[ix, iy] * boundary[ix + 1, iy] == 1:  # Boundary below
            BB[ii, ii] = BB[ii, ii] - 2 / (dy ** 2)
            BB[ii, ii + nx] = BB[ii, ii + nx] * 4 / 3
            if iy != 1:
                BB[ii, ii - nx] = 0
            CC[ii, jj] = CC[ii, jj] + 8 / 3 / (dy ** 4)
            CC[ii, jj + nx + 1] = CC[ii, jj + nx + 1] - 8 / 3 / (dy ** 4)
            CC[ii, jj + 1] = CC[ii, jj + 1] + 8 / 3 / (dy ** 4)
            CC[ii, jj + nx + 2] = CC[ii, jj + nx + 2] - 8 / 3 / (dy ** 4)

        if boundary[ix, iy + 1] * boundary[ix + 1, iy + 1] == 1:  # Boundary above
            BB[ii, ii] = BB[ii, ii] - 2 / (dy ** 2)
            BB[ii, ii - nx] = BB[ii, ii - nx] * 4 / 3
            if iy != ny - 1:
                BB[ii, ii + nx] = 0
            CC[ii, jj] = CC[ii, jj] - 8 / 3 / (dy ** 4)
            CC[ii, jj + nx + 1] = CC[ii, jj + nx + 1] + 8 / 3 / (dy ** 4)
            CC[ii, jj + 1] = CC[ii, jj + 1] - 8 / 3 / (dy ** 4)
            CC[ii, jj + nx + 2] = CC[ii, jj + nx + 2] + 8 / 3 / (dy ** 4)
    print('BB and CC matrix defined')
    CC = CC.tocsr()
    BB = BB.tocsr()
    del AA, Ax, Bx, Ay, By, diagonal_x, diagonal_y, Ix, Iy

    bdata = psi[boundary == 1]
    rhs[1:nx, 1:ny] = -0.25 * (
            vorticity[0:nx - 1, 0:ny - 1] + vorticity[0:nx - 1, 1:ny] + vorticity[1:nx, 0:ny - 1] + vorticity[1:nx, 1:ny])
    rhs[boundary == 1] = bdata

    psi = LU(np.reshape(rhs.T, nxy_one))
    psi = np.reshape(psi.T, (ny + 1, nx + 1)).T
    psi[boundary == 1] = bdata

    ux = 1 / dy * (psi[:, 1:ny + 1] - psi[:, 0:ny])
    uy = -1 / dx * (psi[1:nx + 1, :] - psi[0:nx, :])
    rk = np.array([[0, 1 / 2, 1 / 2, 1], [0, 1, 1, 1], [1, 2, 2, 1]]).T
    ii = 0
    uxs = np.zeros(ux.shape)  # uxs is our stored velocity from our previous time-steps
    uys = np.zeros(uy.shape)  # used to compare our quasi steady state

    t = 0
    resid = np.zeros(10000)  # Defining an array for storage of "residuals"
    pos = 0
    resid[pos] = np.mean((uxs - ux)) / np.mean(ux)

    ux_max = np.max(ux)
    uy_max = np.max(uy)
    dt = np.min([dt_max_diffusion, 1 / (ux_max / dx + uy_max / dy), dt_min])

    temp_value = 0
    storage_steps = int(storage_interval / dt)
    start_time_solver = time.time()
    np.linspace(10, nx-10, 10).astype(int)
    residual_position_x = np.linspace(10, nx-10, 10).astype(int) #np.array((int(nx/4),int(nx/2),int(3*nx/4)))
    residual_position_y = np.linspace(ny-10, 10, 10).astype(int) #np.array((int(3*ny/4),int(ny/2),int(ny/4)))


    print('Time step is {} and steps {}'.format(dt, storage_steps))
    while np.max(np.abs(uxs - ux)) > np.max(uxs) * error and np.max(np.abs(uys - uy)) > np.max(
            uys) * error:  # Finding fluid solution
        ii += 1
        ux_max = np.max(ux)
        uy_max = np.max(uy)
        dt = np.min([dt_max_diffusion, 1 / (ux_max / dx + uy_max / dy),dt_min])

        if ii % storage_steps == 0 and ii - temp_value > 10:
            temp_value = ii
            print('The normalised differences in between the time steps is ', np.max(np.abs(uxs - ux)) / np.max(uxs),
                  np.max(np.abs(uys - uy)) / np.max(uxs), t, time.time() - start_time_solver)
            print(uxs[residual_position_x,residual_position_y] - ux[residual_position_x,residual_position_y])
            print(uys[residual_position_x, residual_position_y] - uy[residual_position_x, residual_position_y])
            pos += 1
            resid[pos] = np.mean(np.absolute(uxs - ux)) / np.mean(uxs)
            uxs = ux
            uys = uy
        for irk in np.arange(4):
            if irk == 0:
                vort1 = vorticity
                psi1 = psi
            else:
                vort1 = vorticity + rk[irk, 0] * dt * d_vorticity_dt1
                rhs[1:nx, 1:ny] = -0.25 * (
                        vort1[0:nx - 1, 0:ny - 1] + vort1[0:nx - 1, 1:ny] + vort1[1:nx, 0:ny - 1] + vort1[1:nx,
                                                                                                    1:ny])
                rhs[boundary == 1] = bdata
                psi1 = LU(np.reshape(rhs.T, nxy_one))
                psi1 = np.reshape(psi1.T, (ny + 1, nx + 1)).T
                psi1[boundary == 1] = bdata

            ux = 1 / dy * (psi[:, 1:ny + 1] - psi[:, 0:ny])
            uy = -1 / dx * (psi[1:nx + 1, :] - psi[0:nx, :])
            fx[1:nx + 1, :] = ux[1:nx + 1, :] * vort1  # Initially assume everything is advected right
            fx[0, :] = ux[0, :] * vort1[0, :]
            reverse = (ux < 0)  # Check where advection is actually leftward
            reverse[nx, :] = 0
            fx[reverse] = ux[reverse] * vort1[reverse[0:nx, :]]  # Recalculate cells where advection is leftward
            fy[:, 1:ny + 1] = uy[:, 1:ny + 1] * vort1  # Initially assume everything is advected up
            fy[:, 0] = uy[:, 0] * vort1[:, 0]
            reverse = (uy < 0)  # Check where advection is actually downward
            reverse[:, ny] = 0
            fy[reverse] = uy[reverse] * vort1[reverse[:, 0:ny]]  # Recalculate cells where advection is downward
            trans_t = (1 / dx) * (fx[1:nx + 1, :] - fx[0:nx, :]) + (1 / dy) * (fy[:, 1:ny + 1] - fy[:, 0:ny])
            intermed = BB.dot(np.reshape(vort1.T, (nxy))) + CC.dot(np.reshape(psi1.T, nxy_one))
            intermed = np.reshape(intermed.T, (ny, nx)).T
            d_vorticity_dt1 = - trans_t + nu * intermed
            d_vorticity_dt2 = rk[irk, 1] * d_vorticity_dt2 + rk[irk, 2] * d_vorticity_dt1

        vorticity = vorticity + (dt / 6) * d_vorticity_dt2
        rhs[1:nx, 1:ny] = -0.25 * (
                vorticity[0:nx - 1, 0:ny - 1] + vorticity[0:nx - 1, 1:ny] + vorticity[1:nx, 0:ny - 1] + vorticity[1:nx, 1:ny])
        rhs[boundary == 1] = bdata

        psi = LU(np.reshape(rhs.T, nxy_one))
        psi = np.reshape(psi.T, (ny + 1, nx + 1)).T
        psi[boundary == 1] = bdata

        ux = 1 / dy * (psi[:, 1:ny + 1] - psi[:, 0:ny])
        uy = -1 / dx * (psi[1:nx + 1, :] - psi[0:nx, :])
        t += dt
    print('elapsed time', (time.time() - start_time) / 60, ' in mins')
    print('Difference between steady state at end of run is', np.max(np.abs(uxs - ux)), np.max(np.abs(uys - uy)))
    print(uxs[residual_position_x, residual_position_y] - ux[residual_position_x, residual_position_y])
    print(uys[residual_position_x, residual_position_y] - uy[residual_position_x, residual_position_y])

    return psi, ux, uy, resid
