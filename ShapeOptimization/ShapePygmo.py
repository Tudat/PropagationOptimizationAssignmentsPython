"""
Copyright (c) 2010-2021, Delft University of Technology
All rights reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

AE4866 Propagation and Optimization in Astrodynamics
Shape Optimization
First name: ***COMPLETE HERE***
Last name: ***COMPLETE HERE***
Student number: ***COMPLETE HERE***

This module is the main script that executes the propagation and optimization. It relies on two other modules, defined
for a more practical organization of functions and classes, which are imported below.

This function computes the dynamics of a capsule re-entering the atmosphere of the Earth, using a
variety of model settings (see below).

The vehicle starts 120 km above the surface of the planet, with a speed of 7.83 km/s in an Earth-fixed frame (see
getInitialState function).

The propagation is terminated as soon as one of the following conditions is met (see 
get_propagation_termination_settings() function):
- Altitude < 25 km
- Propagation time > 24 hr

Only the translational dynamics is propagated. The accelerations are currently set as follows:

0. Aerodynamic acceleration and Earth spherical harmonics gravity up to order 2 and degree 2 (NOMINAL CASE)
1. Aerodynamic acceleration and Earth point mass gravity
2. Aerodynamic acceleration and Earth spherical harmonics gravity up to order 4 and degree 4
3. Lke the nominal case, but the Earth has a simplified rotational model
4. Like the nominal case, but the Earth has a simplified atmosphere model

The trajectory of the capsule is heavily dependent on the shape and orientation of the vehicle. Here, the shape is
determined here by the five parameters, which are used to compute the aerodynamic accelerations on the vehicle using a 
modified Newtonian flow (see also Dirkx and Mooij, 2018). The bank angle and sideslip angles are set to zero.
The vehicle shape and angle of attack are defined by values in the vector shapeParameters.

The entries of the vector 'shapeParameters' contains the following:
- Entry 0:  Nose radius
- Entry 1:  Middle radius
- Entry 2:  Rear length
- Entry 3:  Rear angle
- Entry 4:  Side radius
- Entry 5:  Constant Angle of Attack

The function generating the integrator settings is unchanged with respect to assignment 1. Currently, the integrator
and integrator settings corresponding to the indices 0 and 0 are used, in combination with a Cowell propagator. Use
the combination of integrator and propagator settings that you deem the most suitable, based on the results of
assignment 1.

The script saves the state and dependent variable history for each model settings. In addition, a single file called
'limit_values.dat' is saved with the minimum and maximum time outside which numerical interpolation errors may affect
your comparison. It is up to you whether cutting value outside of that temporal interval during the post-processing.
Finally, outside of the main for loop, the differences in state and dependent variable history with respect to the
nominal case are also written to files. The two are compared at discrete epochs generated with a fixed step size
(see variable output_interpolation_step) valid for all cases. The function compare_models is designed to make
other cross-comparisons possible (e.g. model 2 vs model 3).

The output is written if the variable write_results_to_file is true.
"""

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import os
import numpy as np
import pygmo as pg


# Tudatpy imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.simulation import environment_setup
from tudatpy.kernel.simulation import propagation_setup
from tudatpy.kernel.math import interpolators

# Problem-specific imports
from ShapeOptimizationProblem import ShapeOptimizationProblem, add_capsule_to_body_system
import ShapeOptimizationUtilities as Util

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()
# NOTE TO STUDENTS: INPUT YOUR PARAMETER SET HERE, FROM THE INPUT FILES
# ON BRIGHTSPACE, FOR YOUR SPECIFIC STUDENT NUMBER
shape_parameters = [8.148730872315355,
                    2.720324489288032,
                    0.2270385167794302,
                    -0.4037530896422072,
                    0.2781438040896319,
                    0.4559143679738996]

# Choose whether output of the propagation is written to files
write_results_to_file = False
# Get path of current directory
current_dir = os.path.dirname(__file__)

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Initialize dictionary to save simulation output
simulation_results = dict()

# Set number of models to loop over
number_of_models = 5

# Set the interpolation step at which different runs are compared
output_interpolation_step = 2.0  # s

# Set termination conditions
maximum_duration = constants.JULIAN_DAY  # s
termination_altitude = 25.0E3  # m
# Set vehicle properties
capsule_density = 250.0  # kg m-3
###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################
# Set simulation start epoch
simulation_start_epoch = 0.0  # s
# Load spice kernels
spice_interface.load_standard_kernels()
# Define settings for celestial bodies
bodies_to_create = ['Earth']
# Define coordinate system
global_frame_origin = 'Earth'
global_frame_orientation = 'J2000'
# Create body settings
# N.B.: all the bodies added after this function is called will automatically
# be placed in the same reference frame, which is the same for the full
# system of bodies
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)
# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)
# Create and add capsule to body system
# NOTE TO STUDENTS: When making any modifications to the capsule vehicle, do NOT make them in this code, but in the
# add_capsule_to_body_system function
add_capsule_to_body_system(bodies,
                           shape_parameters,
                           capsule_density)
###########################################################################
# CREATE ACCELERATIONS ####################################################
###########################################################################
# Define bodies that are propagated and their central bodies of propagation
bodies_to_propagate = ['Capsule']
central_bodies = ['Earth']
# Define accelerations for the nominal case
acceleration_settings_on_vehicle = {'Earth': [propagation_setup.acceleration.spherical_harmonic_gravity(2, 2),
                                              propagation_setup.acceleration.aerodynamic()]}
# Create global accelerations' dictionary
acceleration_settings = {'Capsule': acceleration_settings_on_vehicle}
###########################################################################
# CREATE (CONSTANT) PROPAGATION SETTINGS ##################################
###########################################################################
# Retrieve termination settings
termination_settings = Util.get_termination_settings(simulation_start_epoch,
                                                     maximum_duration,
                                                     termination_altitude)
# Retrieve dependent variables to save
dependent_variables_to_save = Util.get_dependent_variable_save_settings()
# Check whether there is any
if not dependent_variables_to_save:
    are_dependent_variables_to_save = False
else:
    are_dependent_variables_to_save = True
# Retrieve initial state
initial_state = Util.get_initial_state(simulation_start_epoch,
                                       bodies)
# Get current propagator, and define translational state propagation settings
current_propagator = propagation_setup.propagator.cowell
# Define propagation settings
current_propagator_settings = propagation_setup.propagator.translational(central_bodies,
                                                                         acceleration_settings,
                                                                         bodies_to_propagate,
                                                                         initial_state,
                                                                         termination_settings,
                                                                         current_propagator,
                                                                         dependent_variables_to_save)
# Note, the following line is needed to properly use the accelerations, and modify them in the Problem class
current_propagator_settings.recreate_state_derivative_models(bodies)
# Create integrator settings
current_integrator_settings = Util.get_integrator_settings(0, 0, 0, simulation_start_epoch)

decision_variable_range = \
    ([ 3.5, 2.0, 0.1, np.deg2rad(-55.0), 0.01, 0.0 ],[ 10.0, 3.0, 5.0, np.deg2rad(-10.0), 0.5, np.deg2rad(30.0) ] )

# Create Lunar Ascent Problem object
current_shape_optimization_problem = ShapeOptimizationProblem(bodies,
                                                              current_integrator_settings,
                                                              current_propagator_settings,
                                                              capsule_density,
                                                              decision_variable_range)



# Select algorithm from pygmo, with one generation
algo = pg.algorithm(pg.de())
# Create pygmo problem
prob = pg.problem(current_shape_optimization_problem)
# Initialize pygmo population with 50 individuals
population_size = 50
pop = pg.population(prob, size=population_size)
# Set the number of evolutions
number_of_evolutions = 50
# Evolve the population recursively
fitness_list = []
population_list = []

fitness_list.append(pop.get_f())
population_list.append(pop.get_x())

for i in range(number_of_evolutions):
    # Evolve the population
    pop = algo.evolve(pop)
    # Store the fitness values for all individuals in a list
    fitness_list.append(pop.get_f())
    population_list.append(pop.get_x())
    print(pop.champion_f)
    print('Evolving population; at generation ' + str(i))



