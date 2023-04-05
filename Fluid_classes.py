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
import os
from pathlib import Path

def loading_data(file_name, folder_names='Empty'):
    """
    :param file_name: Name of loading data file
    :param folder_names: List of folders where file_name is located
    :return: loaded substrate, biomass, mox, mred and current density
    """

    if folder_names == 'Empty':
        loaded_data = np.load(file_name)
        substrate = loaded_data['Acetate']
        biomass = loaded_data['biomass']
        mox = loaded_data['mox']
        mred = loaded_data['mred']
        j = loaded_data['current_density']
        return substrate, biomass, mox, mred, j
    else:
        file_path = Path(os.getcwd())
        for _ in np.arange(len(folder_names)):
            file_path = Path(file_path, folder_names[_])
        file_path = Path(file_path, file_name)
        loaded_data = np.load(file_path)
        substrate = loaded_data['Acetate']
        biomass = loaded_data['biomass']
        mox = loaded_data['mox']
        mred = loaded_data['mred']
        j = loaded_data['current_density']
        return substrate, biomass, mox, mred, j


def save_data_classes(file_name,folder_names, substrate, biomass, mox, mred, j, t):
    """

    :param file_name: name for saved file
    :param folder_names: List of folder where file_name is to be saved
    :param substrate: substrate class
    :param biomass: biomass class
    :param mox: mox class
    :param mred: mred class
    :param j: current density array
    :param t: time class
    :return: will not return anything, instead saves file in location specified
    """

    file_path = Path(os.getcwd())
    for _ in np.arange(len(folder_names)):
        file_path = Path(file_path, folder_names[_])
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = Path(file_path, file_name)
    np.savez_compressed(file_path, Acetate=substrate.current
                        , biomass=biomass.current
                        , mox=mox.current
                        , mred=mred.current
                        , current_density=j
                        , time=t.current)

def loading_data_classes(file_name,folder_names, substrate, biomass, mox, mred, t):
    """

    :param file_name: name for saved file
    :param folder_names: List of folder where file_name is to be saved
    :param substrate: substrate class
    :param biomass: biomass class
    :param mox: mox class
    :param mred: mred class
    :param j: current density array
    :param t: time class
    :return: retuns current density array j, other loaded variables replace current values in classes
    """

    file_path = Path(os.getcwd())
    for _ in np.arange(len(folder_names)):
        file_path = Path(file_path, folder_names[_])
    file_path = Path(file_path, file_name)
    loaded_data = np.load(file_path)
    substrate.current = loaded_data[substrate.name]
    biomass.current = loaded_data['biomass']
    mox.current = loaded_data['mox']
    mred.current = loaded_data['mred']
    t.current = loaded_data['time']
    j = loaded_data['current_density']
    return j

def save_data_classes_two_substrates(file_name,folder_names, substrate, biomass, mox, mred, j, t,substrate2, biomass2,biomass3):
    """
    :param file_name: name for saved file
    :param folder_names: List of folder where file_name is to be saved
    :param substrate: substrate class
    :param biomass: biomass class
    :param mox: mox class
    :param mred: mred class
    :param j: current density array
    :param t: time class
    :return: will not return anything, instead saves file in location specified
    """
    file_path = Path(os.getcwd())
    for _ in np.arange(len(folder_names)):
        file_path = Path(file_path, folder_names[_])
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = Path(file_path, file_name)
    np.savez_compressed(file_path, Acetate=substrate.current
                        , biomass=biomass.current
                        , mox=mox.current
                        , mred=mred.current
                        , current_density=j
                        , time=t.current
                        , substrate2 = substrate2.current
                        , biomass2 = biomass2.current
                        , biomass3 = biomass3.current)

def loading_data_classes_two_substrate(file_name,folder_names, substrate, biomass, mox, mred, t,substrate2, biomass2,biomass3):
    """

    :param file_name: name for saved file
    :param folder_names: List of folder where file_name is to be saved
    :param substrate: substrate class
    :param biomass: biomass class
    :param mox: mox class
    :param mred: mred class
    :param j: current density array
    :param t: time class
    :return: retuns current density array j, other loaded variables replace current values in classes
    """

    file_path = Path(os.getcwd())
    for _ in np.arange(len(folder_names)):
        file_path = Path(file_path, folder_names[_])
    file_path = Path(file_path, file_name)
    loaded_data = np.load(file_path)
    substrate.current = loaded_data[substrate.name]
    biomass.current = loaded_data['biomass']
    mox.current = loaded_data['mox']
    mred.current = loaded_data['mred']
    t.current = loaded_data['time']
    j = loaded_data['current_density']
    substrate2.current = loaded_data['substrate2']
    biomass2.current = loaded_data['biomass2']
    biomass3.current = loaded_data['biomass3']
    return j


class domain:
    def __init__(self, Lx, Ly, Lz, nx=1, ny=1, nz=1, baffle_pairs=0, baffle_length=0):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.biofilm_location = 'Undefined'
        self.volume = self.Lx * self.Ly * self.Lz * 1e3
        self.area = self.Lx * self.Lz

        # self.vol = self.Lx * self.Ly * self.Lz * 1e3
        # self.A = self.Lx * self.Lz

    # def vol(self):
    #     vol = self.Lx * self.Ly * self.Lz * 1e3
    #     return round(vol, 7)

    # def area(self):
    #     area = self.Lx * self.Lz
    #     return round(area, 7)

    def about(self):
        print('All information about the system')
        print('Length x = ', self.Lx, 'm')
        print('Length y = ', self.Ly, 'm')
        print('Length z = ', self.Lz, 'm')
        print('Volume = ', self.Lx * self.Ly * self.Lz * 1e3, 'L')
        print('Area = ', round(self.Lx * self.Lz, 7), 'm^2')
        print('Resolution', self.nx, self.ny)

    def influent_effluent_regions(self, baffle_pairs, baffle_length, width, psi, boundary, flux):
        channel_width = self.Ly / (baffle_pairs * 2 + 1)
        in_out_width = width
        if channel_width > in_out_width:
            in_out_points = int(in_out_width / (self.Ly / self.ny))
        else:  # Default influent region to 1 point if too large
            in_out_points = 1
        in_start = round(int(1 / 2 * self.ny * 1 / (2 * baffle_pairs + 1) - in_out_points / 2 + 1 + 0))
        out_start = self.ny - round(in_out_points / 2) - round(int(1 / 2 * self.ny * 1 / (2 * baffle_pairs + 1) - 0))
        for bb in np.arange(baffle_pairs):  # Determining location for internal walls given length and number
            boundary[1:round(self.nx * baffle_length),
            round(int(2 * bb + 1) * self.ny * 1 / (2 * baffle_pairs + 1)) - 1] = 1
            psi[1:round(self.nx * baffle_length),
            round(int(2 * bb + 1) * self.ny * 1 / (2 * baffle_pairs + 1)) - 1] = flux
            boundary[round(self.nx * (1 - baffle_length) + 1):self.nx,
            round(int(2 * (bb + 1) * self.ny * 1 / (2 * baffle_pairs + 1))) - 1] = 1

        psi[0, 0:round(int(1 / 2 * self.ny * 1 / (2 * baffle_pairs + 1) - in_out_points / 2 + 1))] = 0
        psi[-1, self.ny - round(int(1 / 2 * self.ny * 1 / (2 * baffle_pairs + 1))):self.ny + 1] = flux

        for i in np.arange(in_out_points - 1):  # Influent and effluent width
            psi[0, round(int(
                1 / 2 * self.ny * 1 / (2 * baffle_pairs + 1) - in_out_points / 2 + 1 + i))] = flux / in_out_points * (
                    i + 1)
            psi[-1, self.ny - round(in_out_points / 2) - round(
                int(1 / 2 * self.ny * 1 / (2 * baffle_pairs + 1) - i))] = flux / in_out_points * (i + 1)

        return [psi, boundary, in_out_points, in_start, out_start]


class SystemSetup():

    def __init__(self, length_x, length_y, length_z, nx, ny, nz=1, baffle_pairs=0, baffle_length=0):
        self.biofilm_locations = 0
        self.length_x = length_x
        self.length_y = length_y
        self.length_z = length_z
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.baffle_pairs = baffle_pairs
        self.baffle_length = baffle_length

        self.initialise_system()
        self.define_biofilm_locations()

    def initialise_system(self):
        self.dx = self.length_x / self.nx
        self.dy = self.length_y / self.ny
        self.volume = self.length_x * self.length_y * self.length_z * 1e3
        self.local_volume = self.dx * self.dy * self.length_z * 1e3
        self.local_area = self.dx * self.length_z

        if self.baffle_pairs == 0 or self.baffle_length == 0:
            self.area = self.length_x * self.length_z
        else:
            self.area = self.baffle_length * self.nx * self.local_area

        self.x = np.linspace(0, self.length_x, self.nx).T
        self.y = np.linspace(0, self.length_x, self.ny).T
        [self.yy, self.xx] = np.meshgrid(np.linspace(self.dy / 2, self.length_y - self.dy / 2, self.ny),
                                         np.linspace(self.dx / 2, self.length_x - self.dx / 2, self.nx))

    def define_biofilm_locations(self):
        self.biofilm_locations = 'Function for biofilm locations.'

    def influent_effluent_regions(self, width, flux):
        self.psi = np.zeros((self.nx + 1, self.ny + 1))
        self.boundary = np.zeros((self.nx + 1, self.ny + 1))  # We set this to 1 on all boundary points
        self.boundary[0, :] = 1
        self.boundary[-1, :] = 1
        self.boundary[:, 0] = 1
        self.boundary[:, -1] = 1

        self.psi[0, 0:self.ny + 3] = flux
        self.psi[:, -1] = flux

        # self.boundary[[0, -1], :] = 1
        # self.boundary[:, [0, -1]] = 1

        channel_width = self.length_y / (self.baffle_pairs * 2 + 1)
        in_out_width = width
        if channel_width > in_out_width:
            in_out_points = int(in_out_width / (self.length_y / self.ny))
        else:  # Default influent region to 1 point if too large
            in_out_points = 1
        in_start = round(int(1 / 2 * self.ny * 1 / (2 * self.baffle_pairs + 1) - in_out_points / 2 + 1 + 0))
        out_start = self.ny - round(in_out_points / 2) - round(
            int(1 / 2 * self.ny * 1 / (2 * self.baffle_pairs + 1) - 0))
        for bb in np.arange(self.baffle_pairs):  # Determining location for internal walls given length and number
            self.boundary[1:round(self.nx * self.baffle_length),
            round(int(2 * bb + 1) * self.ny * 1 / (2 * self.baffle_pairs + 1)) - 1] = 1
            self.psi[1:round(self.nx * self.baffle_length),
            round(int(2 * bb + 1) * self.ny * 1 / (2 * self.baffle_pairs + 1)) - 1] = flux
            self.boundary[round(self.nx * (1 - self.baffle_length) + 1):self.nx,
            round(int(2 * (bb + 1) * self.ny * 1 / (2 * self.baffle_pairs + 1))) - 1] = 1

        self.psi[0, 0:round(int(1 / 2 * self.ny * 1 / (2 * self.baffle_pairs + 1) - in_out_points / 2 + 1))] = 0
        self.psi[-1, self.ny - round(int(1 / 2 * self.ny * 1 / (2 * self.baffle_pairs + 1))):self.ny + 1] = flux

        for i in np.arange(in_out_points - 1):  # Influent and effluent width
            self.psi[0, round(int(
                1 / 2 * self.ny * 1 / (
                        2 * self.baffle_pairs + 1) - in_out_points / 2 + 1 + i))] = flux / in_out_points * (
                    i + 1)
            self.psi[-1, self.ny - round(in_out_points / 2) - round(
                int(1 / 2 * self.ny * 1 / (2 * self.baffle_pairs + 1) - i))] = flux / in_out_points * (i + 1)

        # return [psi, self.boundary, in_out_points, in_start, out_start]

    def __repr__(self):
        string_format = (self.__class__.__name__
                         + '\nVolume: {} \n'.format(self.volume)
                         + 'Area: {} \n'.format(self.area)
                         + 'Number of carts: {} \n'.format(2 * self.baffle_pairs)
                         + 'Length of carts (m): {} \n'.format(self.baffle_length * self.length_x)
                         )
        return string_format


class GeneralVariable():
    """
    Class for general parameters,

    initial = Initial conditions for parameter
    
    name = name for parameter, used for saving and loading data
    
    storage_size = 5000, Default number of arrays created to store data during simulation
    
    Returns:
        current = current value for variable
        
        intermediate1 = intermediate value used for first rk4 approximation
        
        intermediate2 = intermediate value used for second rk4 approximation
        
        storage = storage array
    """

    def __init__(self, initial, name, storage_size=5000):
        self.name = name
        self.initial = 1 * initial
        self.current = 1 * initial
        self.intermediate = 0 * initial
        self.storage = self.create_storage(storage_size)
        self.average = 'Mean has not been defined, run update_mean'
        self.ddt1 = 0
        self.ddt2 = 0

    def create_storage(self, storage_size):
        try:
            storage_shape = self.initial.shape
            listing = list(storage_shape)
            listing.append(storage_size)
            storage_shape = tuple(listing)
            storage_array = np.zeros(storage_shape)
            if len(list(self.initial.shape)) == 1:
                """Original array is 1d """
                storage_array[:, 0] = self.initial
            elif len(list(self.initial.shape)) == 2:
                """Original array is 2d """
                storage_array[:, :, 0] = self.initial
            elif len(list(self.initial.shape)) == 3:
                """Original array is 3d """
                storage_array[:, :, :, 0] = self.initial
            return storage_array
        except:
            storage_array = np.zeros(storage_size)
            return storage_array

    def update_mean(self, axis=0):
        self.average = np.mean(self.storage, axis)
        return self.average

    def second_timestep(self, rk_prefactors, itteration):
        self.ddt2 = rk_prefactors[itteration, 1] * self.ddt2 + rk_prefactors[itteration, 2] * self.ddt1
        return self.ddt2

    def update_intermediate(self, rk_prefactors, itteration, dt):
        self.intermediate = self.current + rk_prefactors[itteration, 0] * dt * self.ddt1

    def update_current(self, dt):
        self.current += dt / 6 * self.ddt2

    def about(self, additional='False'):
        """Printing all information about the variable"""
        print('All information about the ' + self.name + ' parameter')
        print('Initial Distribution = \n', self.initial)
        print('Initial Distribution shape is {}'.format(self.initial.shape))
        print('Storage array shape is {}'.format(self.storage.shape))

    def __repr__(self):
        return (self.__class__.__name__ + str('\nName: {}\n').format(self.name))


class MicrobialPopulation(GeneralVariable):
    """
    Class used to define microbial populations growth, decay and consumption rates
    should be given in seconds. second last is monod half rate constant for that species
    finally a name that is given to the species for helpful information about()
    """

    number_of_populations = 0
    population_names = list()

    def __init__(self, initial, consumption, growth, decay_param, sub_monod_coef, name, maximum_mass ,mediator_monod=0, diffusion=0):
        super().__init__(initial, name, storage_size=5000)
        self.max_consumption = consumption  # Max consumption
        self.max_growth = growth  # Max growth
        self.consumption = 0 * consumption  # Current Consumption rates = 0
        self.growth = 0 * growth  # Current Growth rate default = 0
        self.decay = decay_param * self.max_growth
        self.sub_monod_coef = sub_monod_coef  # Substrate monod coefficient
        self.mediator_monod = mediator_monod
        # self.distribution = 'undefined, run positional_distribution(biofilm_location)'
        self.diffusion = diffusion
        self.diffused = 0*self.initial
        self.positional_distribution = 'Undefined, Run: calculate_positional_distribution(bio_loc)'
        self.maximum_mass = maximum_mass
        MicrobialPopulation.number_of_populations += 1
        MicrobialPopulation.population_names.append(self.name)

    def first_timestep(self):
        self.ddt1 = (self.growth - self.decay) * self.intermediate + self.diffused

    def update_growth_and_consumption(self, substrate_local, mediator_local = 'empty'):
        if type(mediator_local) == type('empty'):
            self.growth = self.max_growth * substrate_local / (self.sub_monod_coef + substrate_local)
            self.consumption = self.max_consumption * substrate_local / (self.sub_monod_coef + substrate_local)
        else:
            self.growth = self.max_growth * substrate_local / (self.sub_monod_coef + substrate_local) * mediator_local / (
                    self.mediator_monod + mediator_local)
            self.consumption = self.max_consumption * substrate_local / (
                    self.sub_monod_coef + substrate_local) * mediator_local / (
                                       self.mediator_monod + mediator_local)

    def calculate_positional_distribution(self, biofilm_locations, variation='current'):
        temp = np.zeros(biofilm_locations.shape)
        temp[biofilm_locations == 1] = self.current
        if variation == 'intermediate':
            temp[biofilm_locations == 1] = self.intermediate
        self.positional_distribution = temp
        # self.distribution = np.zeros(biofilm_locations.shape)
        # self.distribution[biofilm_locations == 1] = self.current

    def biomass_diffusion(self, biofilm_locations, diffusion_array):
        """
        Come back to this!
        """
        nx = biofilm_locations.shape[0]
        ny = biofilm_locations.shape[1]
        nxy = nx*ny
        self.calculate_positional_distribution(biofilm_locations, variation='intermediate')
        temp = diffusion_array.dot(np.reshape(self.positional_distribution.T, nxy))
        temp = np.reshape(temp.T, (ny, nx)).T
        temp[biofilm_locations != 1] = 0
        #self.positonal_changes = temp[biofilm_locations==1]
        self.diffused = temp[biofilm_locations == 1]


        #positonal_changes = diffusion_array.dot(positional_dist_temp)
        #return positonal_changes

        # iffusion_array.dot(self.positional_distribution(biofilm_locations))

    def check_mass(self):
        if (self.current > self.maximum_mass).any():
            self.current[(self.current > self.maximum_mass)] -= (self.current[(self.current > self.maximum_mass)] - self.maximum_mass)
        return

    def __repr__(self):
        string_format = (self.__class__.__name__
                         + '\nPopulation: {} \n'.format(self.name)
                         + 'Initial Condition: {} \n'.format(self.initial)
                         + 'Current Mass: {} \n'.format(self.current)
                         + 'Maximum Growth: {} \n'.format(self.max_growth)
                         + 'Maximum Consumption: {} \n'.format(self.max_consumption)
                         )
        return string_format


class Substrate(GeneralVariable):
    """
    Substrate class is to be used for all types of substrate
    """
    item_names = list()

    def __init__(self, initial, influent=-1, diffusion=-1, name=''):
        super().__init__(initial, name, storage_size=5000)
        self.diffusion = diffusion  # only used in the case of substrate
        self.influent = influent
        self.transported = np.zeros(self.initial.shape)
        self.diffused = np.zeros(self.initial.shape)
        self.consumption = np.zeros(self.initial.shape)
        Substrate.item_names.append(self.name)

    def calculate_advection(self, ux, uy, dx, dy):
        nx = ux.shape[0] - 1
        ny = uy.shape[1] - 1
        fx = np.zeros(ux.shape)
        fy = np.zeros(uy.shape)

        fx[1:nx + 1, :] = ux[1:nx + 1, :] * self.intermediate
        fx[0, :] = ux[0, :] * self.intermediate[0, :]
        reverse = (ux < 0)
        reverse[nx, :] = 0
        fx[reverse] = ux[reverse] * self.intermediate[reverse[0:nx, :]]

        fy[:, 1:ny + 1] = uy[:, 1:ny + 1] * self.intermediate  # Initially assume everything is advected up
        fy[:, 0] = uy[:, 0] * self.intermediate[:, 0]
        reverse = (uy < 0)  # Check where advection is actually downward
        reverse[:, ny] = 0
        fy[reverse] = uy[reverse] * self.intermediate[reverse[:, 0:ny]]

        self.transported = (1 / dx) * (fx[1:nx + 1, :] - fx[0:nx, :]) + (1 / dy) * (fy[:, 1:ny + 1] - fy[:, 0:ny])

    def calculate_diffusion(self, comp):
        nx = self.intermediate.shape[0]
        ny = self.intermediate.shape[1]
        nxy = nx * ny
        self.diffused = comp.dot(np.reshape(self.intermediate.T, (nxy)))  # intermediate step for diffusion
        self.diffused = np.reshape(self.diffused.T, (ny, nx)).T  # reshaping back to nx,ny

    def calculate_consumption(self, *populations, biofilm_location, convert_m2_l,biofilm_location2='Empty'):
        bio_number = int(np.sum(biofilm_location))
        self.consumption = np.zeros(self.initial.shape)
        if isinstance(biofilm_location2, str):
            biofilm_location2 = biofilm_location
        glucose_number = int(np.sum(biofilm_location2))
        for n in populations:
            try:
                self.consumption[biofilm_location2 == 1] -= np.reshape(n.consumption * n.intermediate * convert_m2_l,
                                                                  glucose_number)
            except:
                self.consumption[biofilm_location == 1] -= np.reshape(n.consumption * n.intermediate * convert_m2_l,
                                                                      bio_number)
            # print(n.name)

    def first_timestep(self):
        self.ddt1 = (-self.transported) + self.diffusion * self.diffused + self.consumption

    def __repr__(self):
        string_format = (self.__class__.__name__
                         + '\nName: {} \n'.format(self.name)
                         + 'Initial Condition: {} \n'.format(self.initial)
                         + 'Diffusion Coef: {} \n'.format(self.diffusion)
                         )
        return string_format

    def update_influent(self, baffle_pairs, in_out_points, ny, s1=np.zeros(2)):

        """
        Potential Bug fixing required here!!!
        """
        if (s1 == 0).all():
            if self.influent == -1:
                print('Influent ammount is not defined, reverting to original')
                return self.current
            else:
                self.current[0, round(int(1 / 2 * ny * 1 / (2 * baffle_pairs + 1) - in_out_points / 2)):round(int(
                    1 / 2 * ny * 1 / (2 * baffle_pairs + 1) - in_out_points / 2 + 1 + in_out_points))] = self.influent
                return self.current
        else:
            s1[0, round(int(1 / 2 * ny * 1 / (2 * baffle_pairs + 1) - in_out_points / 2)):round(
                int(1 / 2 * ny * 1 / (2 * baffle_pairs + 1) - in_out_points / 2 + 1 + in_out_points))] = s1
            return s1



class FluidPopulation(GeneralVariable):
    """
    FluidPopulation class is to be used for all types of microbial populations exisiting within the fluid and biofilms
    """
    item_names = list()

    def __init__(self, initial, influent=-1, diffusion=-1, name=''):
        super().__init__(initial, name, storage_size=5000)
        self.diffusion = diffusion  # only used in the case of substrate
        self.influent = influent
        self.transported = np.zeros(self.initial.shape)
        self.diffused = np.zeros(self.initial.shape)
        self.consumption = np.zeros(self.initial.shape)
        Substrate.item_names.append(self.name)

    def calculate_advection(self, ux, uy, dx, dy):
        nx = ux.shape[0] - 1
        ny = uy.shape[1] - 1
        fx = np.zeros(ux.shape)
        fy = np.zeros(uy.shape)

        fx[1:nx + 1, :] = ux[1:nx + 1, :] * self.intermediate
        fx[0, :] = ux[0, :] * self.intermediate[0, :]
        reverse = (ux < 0)
        reverse[nx, :] = 0
        fx[reverse] = ux[reverse] * self.intermediate[reverse[0:nx, :]]

        fy[:, 1:ny + 1] = uy[:, 1:ny + 1] * self.intermediate  # Initially assume everything is advected up
        fy[:, 0] = uy[:, 0] * self.intermediate[:, 0]
        reverse = (uy < 0)  # Check where advection is actually downward
        reverse[:, ny] = 0
        fy[reverse] = uy[reverse] * self.intermediate[reverse[:, 0:ny]]

        self.transported = (1 / dx) * (fx[1:nx + 1, :] - fx[0:nx, :]) + (1 / dy) * (fy[:, 1:ny + 1] - fy[:, 0:ny])

    def calculate_diffusion(self, comp):
        nx = self.intermediate.shape[0]
        ny = self.intermediate.shape[1]
        nxy = nx * ny
        self.diffused = comp.dot(np.reshape(self.intermediate.T, (nxy)))  # intermediate step for diffusion
        self.diffused = np.reshape(self.diffused.T, (ny, nx)).T  # reshaping back to nx,ny

    def update_growth_and_consumption(self, substrate_local, mediator_local):
        self.growth = self.max_growth * substrate_local / (self.sub_monod_coef + substrate_local) * mediator_local / (
                self.mediator_monod + mediator_local)
        self.consumption = self.max_consumption * substrate_local / (
                self.sub_monod_coef + substrate_local) * mediator_local / (
                                   self.mediator_monod + mediator_local)


    def first_timestep(self):
        self.ddt1 = (self.growth - self.decay) * self.intermediate + (-self.transported) + self.diffusion * self.diffused

    def __repr__(self):
        string_format = (self.__class__.__name__
                         + '\nName: {} \n'.format(self.name)
                         + 'Initial Condition: {} \n'.format(self.initial)
                         + 'Diffusion Coef: {} \n'.format(self.diffusion)
                         )
        return string_format

    def update_influent(self, baffle_pairs, in_out_points, ny, s1=np.zeros(2)):

        """
        Potential Bug fixing required here!!!
        """
        if (s1 == 0).all():
            if self.influent == -1:
                print('Influent ammount is not defined, reverting to original')
                return self.current
            else:
                self.current[0, round(int(1 / 2 * ny * 1 / (2 * baffle_pairs + 1) - in_out_points / 2)):round(int(
                    1 / 2 * ny * 1 / (2 * baffle_pairs + 1) - in_out_points / 2 + 1 + in_out_points))] = self.influent
                return self.current
        else:
            s1[0, round(int(1 / 2 * ny * 1 / (2 * baffle_pairs + 1) - in_out_points / 2)):round(
                int(1 / 2 * ny * 1 / (2 * baffle_pairs + 1) - in_out_points / 2 + 1 + in_out_points))] = s1
            return s1


class Parameters(GeneralVariable):
    """
    Parameters class is to be used for all other variables including time or mediator. 
    Time has special storage array as this will always be a vector not array
    """
    item_names = list()

    def __init__(self, initial, minimum=-1, maximum=-1, k=-1, name='Not Named!'):
        super().__init__(initial, name, storage_size=5000)
        #     general_variable.__init__(self,initial,name)
        if minimum != -1 or maximum != -1:
            self.minimum = minimum
            self.maximum = maximum
            self.k = k
        Parameters.item_names.append(self.name)

    # def sub_boundary(self,s1 = 0):
    #     if s1 == 0:
    #         if self.influent == -1:
    #             print('Influent ammount is not defined, reverting to original')
    #             return self.now
    #         else:
    #             self.now[0,round(int(1/2*ny*1/(2*baffle_pairs+1)-in_out_points/2)):round(int(1/2*ny*1/(2*baffle_pairs+1)-in_out_points/2+1+in_out_points))] = self.influent
    #             return self.now
    #     else:
    #         s1[0,round(int(1/2*ny*1/(2*baffle_pairs+1)-in_out_points/2)):round(int(1/2*ny*1/(2*baffle_pairs+1)-in_out_points/2+1+in_out_points))] = s1
    #         return s1

def anode_surface_sum(value, location):
    location = np.array(location, dtype=bool)
    temp = np.zeros(location.shape)
    temp[location] = value
    anode_surface_sum = np.sum(temp, 0)
    return anode_surface_sum


def anode_surface_sum_repeated(value, location, additional_pref=1):
    nx = location.shape[0]
    location = np.array(location, dtype=bool)
    temp = anode_surface_sum(value, location)
    anode_total = np.array([additional_pref * temp, ] * nx)
    anode_total = anode_total[location]
    return anode_total


def current_density_inter(E, Rin, Rex, m_total, mred, mox, bio_loc, Ai, R, T, m, F, j0, full=False):
    Rin.current = Rin.minimum
    Rsig = Rin.current + Rex
    eta_conc = R * T / (m * F) * np.log(m_total / mred.intermediate)
    surface_integral_1 = anode_surface_sum_repeated(Ai * mred.intermediate / mox.intermediate * (E.current - eta_conc),
                                                    bio_loc)
    surface_integral_2 = anode_surface_sum_repeated(Ai * mred.intermediate / mox.intermediate, bio_loc,
                                                    additional_pref=Rsig)
    I_anode_repeated = surface_integral_1 / (R * T / (m * F * j0) + surface_integral_2)

    if full:
        for _ in np.arange(10000):
            temp = 1 * I_anode_repeated
            eta_act = E.current - eta_conc - I_anode_repeated * Rsig
            surface_integral_3 = anode_surface_sum_repeated(
                Ai * mred.intermediate / mox.intermediate * np.sinh((m * F) / (2 * R * T) * eta_act), bio_loc)
            surface_integral_4 = anode_surface_sum_repeated(
                Ai * mred.intermediate / mox.intermediate * np.cosh((m * F) / (2 * R * T) * eta_act), bio_loc,
                additional_pref=Rsig)

            if np.isinf(surface_integral_3).any() or np.isinf(surface_integral_4).any():
                intergral_func = 2 * R * T / (m * F) * 1
            else:
                intergral_func = 2 * R * T / (m * F) * ((surface_integral_3 - 1 / (2 * j0) * I_anode_repeated) / (
                        R * T / (m * F * j0) + surface_integral_4))
            I_anode_repeated = I_anode_repeated + intergral_func

            if np.abs(np.mean(temp - I_anode_repeated)) <= 1e-9:
                break

        eta_act = E.current - eta_conc - I_anode_repeated * Rsig
        j = 2 * j0 * np.sinh((m * F) / (2 * R * T) * eta_act) * mred.intermediate / mox.intermediate
        return j, eta_act
    else:
        for rep in np.arange(5):

            temp = 1 * I_anode_repeated
            eta_act = E.current - eta_conc - I_anode_repeated * Rsig
            surface_integral_3 = anode_surface_sum_repeated(Ai * mred.intermediate / mox.intermediate * eta_act,
                                                            bio_loc)
            surface_integral_4 = anode_surface_sum_repeated(Ai * mred.intermediate / mox.intermediate, bio_loc,
                                                            additional_pref=Rsig)
            intergral_func = (surface_integral_3 - R * T / (m * F * j0) * I_anode_repeated) / (
                    R * T / (m * F * j0) + surface_integral_4)
            I_anode_repeated = I_anode_repeated + intergral_func
            if np.abs(np.mean(temp - I_anode_repeated)) <= 1e-10:
                break
        eta_act = E.current - eta_conc - I_anode_repeated * Rsig
        j = j0 * (m * F) / (R * T) * eta_act * mred.intermediate / mox.intermediate
        return j, eta_act


def current_density(E, Rin, Rex, m_total, mred, mox, bio_loc, Ai, R, T, m, F, j0, full=False):
    Rin.current = Rin.minimum
    Rsig = Rin.current + Rex
    eta_conc = R * T / (m * F) * np.log(m_total / mred.current)
    surface_integral_1 = anode_surface_sum_repeated(Ai * mred.current / mox.current * (E.current - eta_conc),
                                                    bio_loc)
    surface_integral_2 = anode_surface_sum_repeated(Ai * mred.current / mox.current, bio_loc,
                                                    additional_pref=Rsig)
    I_anode_repeated = surface_integral_1 / (R * T / (m * F * j0) + surface_integral_2)
    if full:
        for _ in np.arange(10000):
            temp = 1 * I_anode_repeated
            eta_act = E.current - eta_conc - I_anode_repeated * Rsig
            pref = Ai * mred.current / mox.current
            intermed_sinh = pref * np.sinh((m * F) / (2 * R * T) * eta_act)
            intermed_cosh = pref * np.cosh((m * F) / (2 * R * T) * eta_act)
            if np.isinf(intermed_sinh).any() or np.isinf(intermed_cosh).any():
                intergral_func = 2 * R * T / (m * F) * 1
            else:
                sinh_reshaped = anode_surface_sum_repeated(intermed_sinh, bio_loc)
                cosh_reshaped = anode_surface_sum_repeated(intermed_cosh, bio_loc)
                intergral_func = 2 * R * T / (m * F) * (sinh_reshaped - 1 / (2 * j0) * I_anode_repeated) / (
                        R * T / (m * F * j0) + Rsig * cosh_reshaped)
            I_anode_repeated = I_anode_repeated + intergral_func
            if np.abs(np.mean(temp - I_anode_repeated)) <= 1e-9:
                break

        eta_act = E.current - eta_conc - I_anode_repeated * Rsig
        j = 2 * j0 * np.sinh((m * F) / (2 * R * T) * eta_act) * mred.current / mox.current
    else:
        for _ in np.arange(500):
            temp = 1 * I_anode_repeated
            eta_act = E.current - eta_conc - I_anode_repeated * Rsig
            surface_integral_3 = anode_surface_sum_repeated(Ai * mred.current / mox.current * eta_act,
                                                            bio_loc)
            surface_integral_4 = anode_surface_sum_repeated(Ai * mred.current / mox.current, bio_loc,
                                                            additional_pref=Rsig)
            intergral_func = (surface_integral_3 - R * T / (m * F * j0) * I_anode_repeated) / (
                    R * T / (m * F * j0) + surface_integral_4)
            I_anode_repeated = I_anode_repeated + intergral_func
            if np.abs(np.mean(temp - I_anode_repeated)) <= 1e-10:
                break
        eta_act = E.current - eta_conc - I_anode_repeated * Rsig
        j = j0 * (m * F) / (R * T) * eta_act * mred.current / mox.current
    return j, eta_act, eta_conc  # ,j2,temp_j


def mean_effluent(s, in_out_points, out_start):
    temp = np.mean(s.current[-2, out_start:out_start + in_out_points])
    return temp

def advection_multiple(s,g, ux, uy, dx, dy):
    nx = ux.shape[0] - 1
    ny = uy.shape[1] - 1
    fx_2d = np.zeros((ux.shape[0],ux.shape[1],2))
    fy_2d = np.zeros((uy.shape[0],uy.shape[1],2))
    ux_rep = np.repeat(ux[:, :, np.newaxis], 2, axis=2)
    uy_rep = np.repeat(uy[:, :, np.newaxis], 2, axis=2)
    temp_array = np.zeros((nx, ny, 2))
    temp_transported = np.zeros((nx, ny, 2))
    temp_array[:,:,0] = s.intermediate
    temp_array[:,:,1] = g.intermediate

    fx_2d[1:nx + 1, :] = ux_rep[1:nx + 1, :] * temp_array
    fx_2d[0, :,:] = ux_rep[0, :,:] * temp_array[0, :,:]
    reverse = (ux < 0)
    reverse[nx, :] = 0
    fx_2d[reverse,:] = ux_rep[reverse,:] * temp_array[reverse[0:nx, :,],:]

    fy_2d[:, 1:ny + 1,:] = uy_rep[:, 1:ny + 1,:] * temp_array  # Initially assume everything is advected up
    fy_2d[:, 0,:] = uy_rep[:, 0,:] * temp_array[:, 0,:]
    reverse = (uy < 0)  # Check where advection is actually downward
    reverse[:, ny] = 0
    fy_2d[reverse,:] = uy_rep[reverse,:] * temp_array[reverse[:, 0:ny,],:]

    temp_transported = (1 / dx) * (fx_2d[1:nx + 1, :,:] - fx_2d[0:nx, :,:]) + (1 / dy) * (fy_2d[:, 1:ny + 1,:] - fy_2d[:, 0:ny,:])
    s.transported = temp_transported[:, :, 0]
    g.transported = temp_transported[:, :, 1]

    return temp_transported






TC_day = 24 * 60 ** 2
TC_hour = 60 ** 2
TC_min = 60
F = 96485  # As/K Faraday's const
R = 8.314472  # Gas const
T = 298  # K

if __name__ == '__main__':
    rk = np.array([[0, 1 / 2, 1 / 2, 1], [0, 1, 1, 1], [1, 2, 2, 1]]).T
    print('Code contains functions responsable for finding steady state of velocity, etc')
    Za = MicrobialPopulation(10 * np.ones((10)), 7.9 / TC_day, 0.7 / TC_day, 0, 80, 'Anodophilic',
                             mediator_monod=0.02 * 0.05)
    domain = SystemSetup(0.32, 0.45, 0.25, 300, 300, 1, baffle_pairs=5, baffle_length=91 / 100)
    # Za.about()
    s = Substrate(500 * np.ones((1, 1)), name='Acetate')
    # s.about()
    # Za.storage[:,1] = Za.storage[:,0]*10
    # print(Za.average)
    # Za.update_mean(0)
    # print(Za.average)
    # print(Za.ddt1,Za.second_timestep(rk,1))
    t = GeneralVariable(0, name='Time')
    print(Za)
    print(t)

    flux = (domain.length_x * domain.length_y) / (72 * TC_hour)
    domain.influent_effluent_regions(domain.dy * 18, flux)

    t
    domain
