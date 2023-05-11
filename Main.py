# -*- coding: utf-8 -*-

# Copyright 2020 Jordan Day
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


# import numpy as np
# import scipy as sp
from scipy import sparse
# from scipy.sparse import linalg
# import matplotlib.pyplot as plt
# import time
from tqdm import tqdm
from Fluid_classes import *
from Plotting_functions import *
import os
from pathlib import Path
import solver

np.random.seed(0)
nx = 300
ny = 300
nz = 1
Lx = 0.32  # length in m
Ly = 0.45  # length in m
Lz = 0.25  # length in m
hrt = 24  # hrt set in any given time units TC is the conversion to seconds
file_name = 'HRT_{}_ym_26_Three_Population_area_optimisation_side_'.format(int(hrt))  # File name to save output data

hrt *= TC_hour  # The hydraulic retention time converted into seconds
baffle_length = 91 / 100  # This is the fraction of the tank a baffle takes up in x
baffle_pairs = 5  # Number of baffle pairs (RHS+LHS) = 1 pair.

# Baffle param
rho = 8000
# Runtime for reactor system
RT = 1
RT *= TC_day
dt_max = 8
k = 20  # Store data every k min, 1 is every min, 2 every 2 min
D = 1 / hrt
anode_side = 'all'  # can either be 'all', 'influent', 'effluent' 'front_loaded' based on the layout you wish to test
file_name += anode_side
print(file_name)

dx = Lx / nx
dy = Ly / ny
x = np.linspace(0, Lx, nx).T
y = np.linspace(0, Ly, ny).T
[yy, xx] = np.meshgrid(np.linspace(dy / 2, Ly - dy / 2, ny), np.linspace(dx / 2, Lx - dx / 2, nx))

system = domain(Lx, Ly, Lz, nx=nx, ny=ny)

flux = (Lx * Ly) / hrt  # The "area flux" through the system
nxy = nx * ny
nxy_one = (nx + 1) * (ny + 1)
psi = np.zeros((nx + 1, ny + 1))  # This is the stream function
boundary = np.zeros((nx + 1, ny + 1))  # We set this to 1 on all boundary points
boundary[0, :] = 1
boundary[-1, :] = 1
boundary[:, 0] = 1
boundary[:, -1] = 1

edges = boundary
psi[0, 0:ny + 3] = flux
psi[:, -1] = flux
psi, boundary, in_out_points, in_start, out_start = system.influent_effluent_regions(baffle_pairs, baffle_length,
                                                                                     dy * 18, psi, boundary, flux)

bdata = psi[boundary == 1]

file_a = 'hrt' + str(hrt).replace('.', '_') + '_nx' + str(nx) + '_ny' + str(ny) + '_Lx' + str(Lx) + '_Ly' + str(
    Ly) + '_pairs' + str(baffle_pairs) + '_width' + str(np.round(baffle_length, decimals=1)).replace('.',
                                                                                                     '_') + '.csv'
file_x = 'Ux_' + file_a
file_y = 'Uy_' + file_a
data_folder = Path(os.getcwd(),'Velocity Fields')

try:
    ux = np.genfromtxt(data_folder / file_x, delimiter=',')
    uy = np.genfromtxt(data_folder / file_y, delimiter=',')
    print('Velocity Files loaded from text files')
except:
    psi, ux, uy, resid = solver.steady_state(boundary, psi, nx, ny, dx, dy,error=1e-6)  # Using function to determine steady state
    save_velocity_fields(file_a,['Velocity Fields'],ux,uy)
    print('Velocity fields have been determined and saved in text files \n')

# %%
external = np.zeros(boundary.shape)
external[0, :] = 1
external[-1, :] = 1
external[:, 0] = 1
external[:, -1] = 1
internal = boundary - external
bio_loc = np.zeros(boundary.shape)
bio_loc[:, 0:ny] += internal[:, 1:ny + 1]
bio_loc[:, 1:ny + 1] += internal[:, 0:ny]
bio_loc = bio_loc[0:nx, 0:ny]
if baffle_length == 0 and baffle_pairs == 0:
    bio_loc[1:-1, -2] = 1
    bio_loc[1:-1, 1] = 1

positional = np.nonzero(np.mean(bio_loc, 0))
switch = np.zeros(bio_loc.shape)
if anode_side == 'influent':
    glucose_loc = 1*bio_loc
    switch[:, positional[0][0:20:2]] = 1  # :2 to alternate front and back on off 1:20:2 is back, 0:20:2 is front
    bio_loc *= switch
    print('influent facing anodes are active')
elif anode_side == 'effluent':
    glucose_loc = 1*bio_loc
    switch[:, positional[0][1:20:2]] = 1  # :2 to alternate front and back on off 1:20:2 is back, 0:20:2 is front
    bio_loc *= switch
    print('effluent facing anodes are active')
elif anode_side == 'front_loaded':
    switch[:, positional[0][10:20:1]] = 1
    glucose_loc = 1 * bio_loc
    bio_loc *= switch
    #glucose_loc = glucose_loc - bio_loc
    print('First 5 carts are fermenting only')
else:
    anode_side = 'all'
    #switch[:, positional[0][11:20:1]] = 1
    #bio_loc *= switch
    glucose_loc = 1 * bio_loc
    print('All anodes are active')

bio_number = np.count_nonzero(bio_loc)
glucose_number = np.count_nonzero(glucose_loc)
# %%

anode_numbers = np.count_nonzero(np.mean(bio_loc, 0))

# Determine anode area based on biofilm and baffle length
Ai = dx * Lz
A = baffle_length * nx * Ai
# A = Lx * Lz #anode area
# Ai = A/nx # area per cell

Vol = Lx * Ly * Lz * 1e3  # Volume in
Voli = dx * dy * Lz * 1e3  # Local volume in L

convert_m2_l = Ai / Voli
convert_l_m2 = Voli / Ai

z1 = MicrobialPopulation(5000 * np.ones(bio_number)  # np.random.normal(loc = 1000,scale = 10,size = (nx,2)) #initial
                         , 7.9 / TC_day  #7.9 / TC_day  # consumption
                         , 2*0.7 / TC_day  # growth
                         , 0.02  # decay
                         , 20 # sub monod const
                         , 'Anodophilic'  # Defining anodophilic species as class
                         , 5000
                         , mediator_monod=0.2 * 1)

z3 = MicrobialPopulation(512 * np.ones(glucose_number)
                         , 8.3 / TC_day
                         , 2*0.3 / TC_day
                         , 0.02
                         , 80
                         , 'Methanogenic',
                         5000)

z2 = MicrobialPopulation( 2000 * np.ones(glucose_number)
                         , 4.8/ TC_day
                         , 2*0.4 / TC_day
                         , 0.02
                         , 100
                         , 'Glucose Consumer'
                         , 2000)

# Doubles initial bacteria conc on biofilms without electrogens
z2_space = 0*glucose_loc
z2_space[glucose_loc == 1] = z2.current
z2_space[glucose_loc - bio_loc == 1] = 4000
z2.current = z2_space[glucose_loc==1]

z3_space = 0*glucose_loc
z3_space[glucose_loc == 1] = z3.current
z3_space[glucose_loc - bio_loc == 1] = 2*z3.current.max()
z3.current = z3_space[glucose_loc==1]


s = Substrate(100 * np.ones((nx, ny)), influent=150, diffusion=1e-9, name='Acetate')
s.current = s.update_influent(baffle_pairs, in_out_points, ny)
s2 = Substrate(100* np.ones((nx, ny)),influent= 1500,diffusion= 1e-9,name = 'Glucose') # 300 or 1500
s2.current = s2.update_influent(baffle_pairs, in_out_points, ny)
s2.s_yield = 0.7 ################################################### Latest run using 0.8 as Ac yield

m_total = 1  # mg-M / mg-Z Initial mediator

mox = Parameters(0.99 * m_total * np.ones(bio_number), name='Oxidised Mediator')

mred = Parameters(m_total - mox.initial, name='Reduced Mediator')
Ym = 26#22.75  # mg-M /mg -s 36#32
m = 2  # mol-e / mol-M
gamma = 663400  # mg-M/mol-M
T = 298  # K
j0 = 1#e-2  # 1e-2-> almost identical to 1, but much faster run times
BV_full = False

j_min = 1.60
j_max = 1.60  # 1.34#0.64
##############################################
# changed here and j_0 ###
E_min_scaled = j_min * (A * 500)  # /(R*T/(m*F*j0))Anode area times sum of res
E_max_scaled = j_max * (A * 500)  # /(R*T/(m*F*j0)) Anode area times sum of res

#############################################
j_test = 1.5  # .4#2.4#1.6#2.4/10  # 40 j0 = 1e-4
# Full BV stable for hrt = 2, 6, j0 = 1e-4 , J_test = 1.4

Rin = Parameters(0, minimum=7, maximum=7, k=0.006 * A / Vol, name='Internal Resistance')
Rex = 1  # ohm
E_test = 0.8  # j_test * (A * (Rin.minimum+Rex)) # j_test*(R*T/(m*F*j0)+500*A*(0.92/0.08)/(0.92/0.08))

E = Parameters(0, minimum=E_test, maximum=E_test, k=0.0006, name='Voltage')
# E = Parameters(0, minimum=10, maximum=10, k=0.0006, name='Voltage')

# E_ocv = E_test * bio_loc
# E_ocv[:,270:273:2] = 0.9*(A * (Rin.minimum+Rex))
# E_ocv = E_ocv[bio_loc == 1]
# E.current = E_ocv


pref = gamma / (m * F)
I = Parameters(0, name='Current (Not to be confused with current value)')
s_blank = np.zeros((bio_number, 5000))
ij = 0
t = GeneralVariable(0, name='Time')
Rin.current = Rin.minimum  # +(Rin.maximum - Rin.minimum)*np.exp(-Rin.k*sum(z1.initial)/nx) + Rex
Rin.storage[0] = Rin.current

# setting up locations for biofilm
# bio_loc = np.zeros((nx,ny))
bio_upper = np.zeros((nx, ny))
bio_lower = np.zeros((nx, ny))
bio_lower[:, -2] = 1  # used for plotting
bio_upper[:, 1] = 1
consump = np.zeros((nx, ny))  # setting consumption array
med_dist = np.zeros(consump.shape)

ux_max = np.max(ux)  # max velocity based on steady state
uy_max = np.max(uy)  # max vel from steady state
# Creating sparse matrix for biofiolm diffusion]
positions = [-1, 0, 1]
diag_x = np.array([[1 / (dx ** 2)], [-2 / (dx ** 2)], [1 / (dx ** 2)]]).repeat(nx, axis=1)
diag_y = np.array([[1 / (dy ** 2)], [-2 / (dy ** 2)], [1 / (dy ** 2)]]).repeat(ny, axis=1)
Dx = sp.sparse.spdiags(diag_x, positions, nx, nx)  # d/dx mat Alternate approach to using diffuse_S array
Dy = sp.sparse.spdiags(diag_y, positions, ny, ny)  # d/dy mat
kappa_bio = 1e-12  # diffusion rate for biofilm
Dx_bio = sp.sparse.spdiags(diag_x, positions, nx, nx).tolil()
Dx_bio[0, -1] += 1 / (dx ** 2)  # Periodic Boundary Conditions
Dx_bio[-1, 0] += 1 / (dx ** 2)  # Periodic BC
# Dx_bio[0,-1] += 1/(dx**2) # non peridoic bc
# Dx_bio[0,-1] += 1/(dx**2) # Need to fix
Dx_bio *= kappa_bio  # setting up diffusion array for biofilm
Dx_bio = Dx_bio.tocsr()

Bx = sp.sparse.spdiags(diag_x, positions, nx, nx)  # tolil()
By = sp.sparse.spdiags(diag_y, positions, ny, ny)
Iy = sp.sparse.eye(ny)
Ix = sp.sparse.eye(nx)
Diffuse_s = (sp.sparse.kron(Iy, Bx) + sp.sparse.kron(By, Ix)).tolil()

bio_diffusion_x = sp.sparse.kron(Iy, Bx).tolil()
temp_location = np.zeros((nx + 1, ny + 1))
temp_location[:-1, :-1] = bio_loc

for ii in np.arange(nxy):
    ix = int(ii % nx)
    iy = int(np.floor(ii / nx))
    jj = iy * (nx + 1) + ix
    if boundary[ix, iy] * boundary[ix, iy + 1] == 1:  # Boundary on left
        Diffuse_s[ii, ii] = Diffuse_s[ii, ii] + 1 / (dx ** 2)  #
        if ix != 0:
            Diffuse_s[ii, ii - 1] = 0
    if boundary[ix + 1, iy] * boundary[ix + 1, iy + 1] == 1:  # Boundary on right
        Diffuse_s[ii, ii] = Diffuse_s[ii, ii] + 1 / (dx ** 2)
        if ix != nx - 1:
            Diffuse_s[ii, ii + 1] = 0

    if boundary[ix, iy] * boundary[ix + 1, iy] == 1:  # Boundary below
        Diffuse_s[ii, ii] = Diffuse_s[ii, ii] + 1 / (dy ** 2)
        if iy != 0:
            Diffuse_s[ii, ii - nx] = 0

    if boundary[ix, iy + 1] * boundary[ix + 1, iy + 1] == 1:  # Boundary above
        Diffuse_s[ii, ii] = Diffuse_s[ii, ii] + 1 / (dy ** 2)
        if iy != ny - 1:
            Diffuse_s[ii, ii + nx] = 0
    if temp_location[ix, iy] * temp_location[ix + 1, iy] == 0:
        bio_diffusion_x[ii, ii] += 1 / (dx ** 2)
    if ix == 0:
        bio_diffusion_x[ii, ii] += 1 / (dx ** 2)
    if ix != 0:
        if temp_location[ix - 1, iy] * temp_location[ix, iy] == 0:
            bio_diffusion_x[ii, ii] += 1 / (dx ** 2)

    if temp_location[ix, iy] + temp_location[ix + 1, iy] == 0:
        bio_diffusion_x[ii, ii] -= 1 / (dx ** 2)

Diffuse_s = Diffuse_s.tocsr()
LU = sp.sparse.linalg

bio_diffusion_x = 0*1e-9 * bio_diffusion_x.tocsr()
z1.biomass_diffusion(bio_loc,bio_diffusion_x)
z2.biomass_diffusion(glucose_loc,bio_diffusion_x)
z3.biomass_diffusion(glucose_loc,bio_diffusion_x)



z1.calculate_positional_distribution(bio_loc)
temp = bio_diffusion_x.dot(np.reshape(z1.positional_distribution.T, nxy))
temp = np.reshape(temp.T, (ny, nx)).T
temp[bio_loc != 1] = 0  # Deals with mass produced outside of biofilm! (temp fix)

del temp_location

s.storage[:, :, 0] = s.initial
s.current[:, :] = s.initial  # This line causes s.initial to be linked to s.now

dt = min(dt_max, 1 / (ux_max / dx + uy_max / dy), (dx ** 2 * dy ** 2) / (
        2 * s.diffusion * (dx ** 2 + dy ** 2)))
# dt = np.floor(dt*100)/100

ii = 0
rk = np.array([[0, 1 / 2, 1 / 2, 1], [0, 1, 1, 1], [1, 2, 2, 1]]).T
bound_out = np.zeros(s.current.shape)
bound_in = np.zeros(s.current.shape)
bound_out[-1, :] = 1
bound_out[:, -1] = 1
bound_out[:, 0] = 1
bound_in[-2, :] = 1
bound_in[:, -2] = 1
bound_in[:, 1] = 1

E.current = E.minimum
Rin.current = Rin.minimum
Rsig = Rin.current + Rex

mred.current = m_total - mox.current
eta_conc = R * T / (m * F) * np.log(m_total / mred.current)

med_dist[bio_loc == 1] = Ai * mred.current / mox.current
summation = np.array([Rsig * np.sum(med_dist, 0), ] * nx)
summation_shaped = summation[bio_loc == 1]
j = (mred.current / mox.current * (E.current - eta_conc)) / (
        R * T / (m * F * j0) + summation_shaped)
del eta_conc

print("System will simulate a {} baffle system with a fluid HRT of {} hours and bio runtime of {} days".format(
    2 * baffle_pairs, hrt / TC_hour, RT / TC_day))

start_time_bio = time.time()
storage_steps = int(k * 60 / dt)
z1.diffused *= 0
z2.diffused *= 0
z2.diffused *= 0
combined_max = 8000
total_time = time.time()
pbar = tqdm(total=101, desc="Progress of simulation", ncols=100, )

fermenting_bio_loc = glucose_loc - bio_loc

while t.current < RT + 10:
    ii += 1
    lt = time.time()
    irk = 0
    consump *= 0  # Reset consumption
    while irk < 4:  # replaced with while loop to allow for adaptive timesteps
        if irk == 0:
            z1.intermediate = z1.current
            z3.intermediate = z3.current
            s.intermediate = s.current
            mox.intermediate = mox.current
            mred.intermediate = m_total - mox.intermediate

            z2.intermediate = z2.current
            s2.intermediate = s2.current
        else:
            z1.update_intermediate(rk, irk, dt)
            z3.update_intermediate(rk, irk, dt)
            s.update_intermediate(rk, irk, dt)
            mox.update_intermediate(rk, irk, dt)
            mred.intermediate = m_total - mox.intermediate

            s.intermediate[0, round(int(1 / 2 * ny * 1 / (2 * baffle_pairs + 1) - in_out_points / 2)):round(
                int(1 / 2 * ny * 1 / (2 * baffle_pairs + 1) - in_out_points / 2 + 1 + in_out_points))] = s.influent

            z2.update_intermediate(rk, irk, dt)
            s2.update_intermediate(rk, irk, dt)
            s2.intermediate[0, round(int(1 / 2 * ny * 1 / (2 * baffle_pairs + 1) - in_out_points / 2)):round(
                int(1 / 2 * ny * 1 / (2 * baffle_pairs + 1) - in_out_points / 2 + 1 + in_out_points))] = s2.influent



        if (mox.current + (dt / 6) * mox.ddt2 > m_total).any() or (
                mox.current + rk[irk, 0] * dt * mox.ddt1 > m_total).any():
            # If over estimating rk4 loop is reset with smaller timestep
            irk = 0
            dt *= 0.5
            continue


        local_g = np.reshape(s2.intermediate[glucose_loc == 1], (glucose_number))
        z2.update_growth_and_consumption(local_g)
        z2.first_timestep()
        z2.second_timestep(rk, irk)

        local_s = np.reshape(s.intermediate[bio_loc == 1], bio_number)
        local_s_meth = np.reshape(s.intermediate[glucose_loc == 1], (glucose_number))

        substrate_mean_surface = anode_surface_sum(Ai * local_s, bio_loc) / A

        j, eta_act = current_density_inter(E, Rin, Rex, m_total, mred, mox, bio_loc, Ai, R, T, m, F, j0, full=BV_full)
        I_anode = anode_surface_sum_repeated(j * Ai, bio_loc)
        mediator_current_density_term = I_anode / A

        # local_s = np.reshape(s.intermediate[bio_loc == 1], (bio_number))
        z1.update_growth_and_consumption(local_s, mox.intermediate)
        z3.update_growth_and_consumption(local_s_meth)
        # diff_z1 = 0  # *Dx_bio.dot(vdata1[0,])
        # diff_z3 = 0  # *Dx_bio.dot(vdata1[1,]) # diffusion of X and Z
        z1.biomass_diffusion(bio_loc, bio_diffusion_x)
        z3.biomass_diffusion(glucose_loc, bio_diffusion_x)
        z2.biomass_diffusion(glucose_loc, bio_diffusion_x)
        z1.first_timestep()
        z1.second_timestep(rk, irk)
        z3.first_timestep()
        z3.second_timestep(rk, irk)
        mox.ddt1 = -Ym * z1.consumption + pref * j / z1.intermediate
        mox.second_timestep(rk, irk)
        s.calculate_advection(ux, uy, dx, dy)  # Advection is slowest process
        s.calculate_diffusion(Diffuse_s)  # diff is second slowest
        s.calculate_consumption(z1, z3, biofilm_location=bio_loc, convert_m2_l=convert_m2_l,biofilm_location2=glucose_loc)  # rapid
        fluid_time = time.time()
        s.first_timestep()  # Timestepping is almost as slow as diff
        s.created = -s2.s_yield * s2.consumption
        s.ddt1 += s.created
        s.second_timestep(rk, irk)
        s2.calculate_advection(ux, uy, dx, dy)
        s2.calculate_diffusion(Diffuse_s)
        s2.calculate_consumption(z2, biofilm_location=glucose_loc, convert_m2_l=convert_m2_l)
        s2.first_timestep()
        s2.second_timestep(rk, irk)
        irk += 1  # move forward in rk4 loop
        if irk == 4 and (mox.current + (dt / 6) * mox.ddt2 > m_total).any():
            irk = 0
            dt *= 0.5
            # print('Loop restart')
            continue

    z2.update_current(dt)
    s2.update_current(dt)

    z1.update_current(dt)
    z3.update_current(dt)
    s.update_current(dt)
    mox.update_current(dt)
    mred.current = m_total - mox.current

    j, eta_act, eta_conc = current_density(E, Rin, Rex, m_total, mred, mox, bio_loc, Ai, R, T, m, F, j0, full=BV_full)
    I.current = np.sum(Ai * j)
    t.current += dt
    dt = min(dt_max, dt * 2, 1 / (ux_max / dx + uy_max / dy), (dx ** 2 * dy ** 2) / (
            2 * s.diffusion * (dx ** 2 + dy ** 2)))  # increase timestep up to 2 or double previous timestep
    s.current = s.update_influent(baffle_pairs, in_out_points, ny)
    z2.calculate_positional_distribution(glucose_loc)
    z3.calculate_positional_distribution(glucose_loc)
    temp_shared1 = z2.positional_distribution[bio_loc == 1]
    temp_shared2 = z3.positional_distribution[bio_loc == 1]
    total_biomass = z1.current + temp_shared1 + temp_shared2#+ z2.current
    if (total_biomass > rho).any():
        z1.current[total_biomass > rho] *= rho/total_biomass[total_biomass > rho]
        temp_shared1[total_biomass > rho] *= rho / total_biomass[total_biomass > rho]
        temp_shared2[total_biomass > rho] *= rho / total_biomass[total_biomass > rho]
        z2.positional_distribution[bio_loc == 1] = temp_shared1
        z3.positional_distribution[bio_loc == 1] = temp_shared2


    temp_mass1 = z2.positional_distribution[fermenting_bio_loc == 1]
    temp_mass2 = z3.positional_distribution[fermenting_bio_loc == 1]
    total_biomass2 = temp_mass1 + temp_mass2
    if (total_biomass2 > rho).any():
        temp_mass1[total_biomass2 > rho] *= rho / total_biomass2[total_biomass2 > rho]
        temp_mass2[total_biomass2 > rho] *= rho / total_biomass2[total_biomass2 > rho]
        z2.positional_distribution[fermenting_bio_loc == 1] = temp_mass1
        z3.positional_distribution[fermenting_bio_loc == 1] = temp_mass2

    z2.current = z2.positional_distribution[glucose_loc == 1]
    z3.current = z3.positional_distribution[glucose_loc == 1]
    
    if ii % storage_steps == 0 or ii == 1:  # round(t.now,2)%(k*60) == 0 : #Storage of data
        ij += 1
        z1.storage[:, ij] = z1.current
        z3.storage[:, ij] = z3.current
        mox.storage[:, ij] = mox.current
        mred.storage[:, ij] = mred.current
        I.storage[ij] = I.current
        t.storage[ij] = t.current
        s.storage[:, :, ij] = s.current
        s_blank[:, ij] = z1.storage[:,ij-1] - z1.storage[:,ij]  # (muz-Kda)*vdata[0,]
        z2.storage[:,ij] = z2.current
        s2.storage[:,:,ij] = s2.current

        increase = round((t.current - t.storage[ij - 1]) / (RT + 20) * 100, 1)
        pbar.update(round(increase, 1))

#

gl_removal_percent = (s2.influent - mean_effluent(s2,in_out_points,out_start))/s2.influent*100
s_removal_percent = (s.influent - mean_effluent(s,in_out_points,out_start))/s.influent*100
influent_cod = s2.influent*1.07+s.influent*1.06
effluent_cod = mean_effluent(s2,in_out_points,out_start)*1.07 + mean_effluent(s,in_out_points,out_start)*1.06
cod_removal_percent = (influent_cod - effluent_cod)/influent_cod*100
print('\n HRT: {} hours\n Glucose removal percent: {} \n Acetate removal percent: {} \n COD removal percent: {} \n Current Density: {} \n'.format(hrt/TC_hour,
    gl_removal_percent,s_removal_percent,cod_removal_percent,j.mean()))

#save_data_classes_two_substrates(file_name,['Output','temp'],s,z1,mox,mred,j,t,s2,z2)
save_data_classes_two_substrates(file_name,['Output','Three_Population_Files'],s,z1,mox,mred,j,t,s2,z2,z3)

z1.update_mean(0)
z2.update_mean(0)
z3.update_mean(0)
plot_time_series(t.storage[0:ij+1]/TC_day,np.trim_zeros(z2.average),sty = 'r--',linelabel = 'Glucose Consumer',new_fig=True)
plot_time_series(t.storage[0:ij+1]/TC_day,np.trim_zeros(z1.average),sty = 'g-',linelabel = 'Anodophilic',xlab='Time (Days)', ylab= 'Biomass density',title= 'Anodophilic,Fermentor and Methanogen' )
plot_time_series(t.storage[0:ij+1]/TC_day,np.trim_zeros(z3.average),sty = 'k-',linelabel = 'Methanogen',xlab='Time (Days)', ylab= 'Biomass density',title= 'Anodophilic,Fermentor and Methanogen' )
plot_time_series(t.storage[0:ij+1]/TC_day,np.trim_zeros(z2.average+z1.average+z3.average),sty = 'b.--',linelabel = 'Total',xlab='Time (Days)', ylab= 'Biomass density',title= 'Anodophilic,Fermentor and Methanogen' )

print(t.current)
plot_positional_data(x, j, bio_loc, new_fig=True)

print(current_density_inter(E, Rin, Rex, m_total, mred, mox, bio_loc, Ai, R, T, m, F, j0, full=False))
current_density_inter(E, Rin, Rex, m_total, mred, mox, bio_loc, Ai, R, T, m, F, j0, full=True)

print(dt, t.current)
plot_positional_data(x, j, bio_loc, new_fig=True, side='Left',
                     title='Current density using linear BV Eocv = {}'.format(E.current))

plot_positional_data(x, eta_conc, bio_loc, side='left', new_fig=True,
                     title='$\eta_\mathrm{conc} = M_\mathrm{total}/M_\mathrm{red}$')

plt.figure(figsize=(14, 10))
plt.subplot(221)
plot_contour(xx, yy, s.current)
plt.subplot(222)
plot_positional_data(x, j, bio_loc, side='right', title='Positional current density A/m^2',ylab = 'Current Density (A/m^2)',xlab = 'x')
plt.subplot(223)
plot_time_series(t.storage[0:ij + 1] / TC_day, I.storage[0:ij + 1] / (20 * A), linelabel='Current density over time',ylab = 'Current Density (A/m^2)',xlab = 'Time (Days)')
plt.subplot(224)
plot_positional_data(x, z1.current, bio_loc, side='right', title='Positional biomass mg/m^2',ylab = 'Anodophilic Density (mg/m^2)',xlab = 'x')
