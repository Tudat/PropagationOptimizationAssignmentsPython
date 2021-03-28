"""
Copyright (c) 2010-2021, Delft University of Technology
All rights reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

AE4866 Propagation and Optimization in Astrodynamics
Lunar Ascent
First name: ***COMPLETE HERE***
Last name: ***COMPLETE HERE***
Student number: ***COMPLETE HERE***

This module is the main script that executes the propagation and optimization. It relies on two other modules, defined
for a more practical organization of functions and classes, which are imported below.

This function computes the dynamics of a lunar ascent vehicle, starting at zero velocity on the Moon's surface.
 
The propagation starts near the lunar surface, with a speed relative to the Moon of 10 m/s.

The propagation is terminated as soon as one of the following conditions is met:

* Altitude > 100 km
* Altitude < 0 km
* Propagation time > 3600 s
* Vehicle mass < 2250 kg

Both the translational dynamics and mass of the vehicle are propagated, using a fixed specific impulse.
Only the thrust acceleration is currently included. It is up to you to modify the dynamical model based on the results
of assignment 2.

The thrust is computed based on a fixed thrust magnitude, and a variable thrust direction. The trust direction is defined
on a set of 5 nodes, spread evenly in time. At each node, a thrust angle theta is defined, which gives the angle between
the -z and y angles in the ascent vehicle's vertical frame (see Mooij, 1994). Between the nodes, the thrust is linearly
interpolated. If the propagation goes beyond the bounds of the nodes, the boundary value is used. The thrust profile
is parameterized by the values of the vector thrustParameters.

The entries of the vector 'thrustParameters' contains the following:
- Entry 0: Constant thrust magnitude
- Entry 1: Constant spacing in time between nodes
- Entry 2-6: Thrust angle theta, at nodes 1-5 (in order)

The function generating the integrator settings is unchanged with respect to assignment 1. Currently, the integrator
and integrator settings corresponding to the indices 0 and 0 are used, in combination with a Cowell propagator. Use
the combination of integrator and propagator settings that you deem the most suitable, based on the results of
assignment 1.

The last part of the code runs the optimization problem with PyGMO. For more information, see:
https://tudat-space.readthedocs.io/en/latest/_src_examples/pygmo_basics.html.
"""

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np
import os
import pygmo as pg

# Tudatpy imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.simulation import environment_setup
from tudatpy.kernel.simulation import propagation_setup
from tudatpy.kernel.math import interpolators

# Problem-specific imports
from LunarAscentProblem import LunarAscentProblem
from LunarAscentProblem import get_thrust_acceleration_model_from_parameters
import LunarAscentUtilities as Util

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()
# Define problem independent variables
thrust_parameters = [15629.13262285292,
                     21.50263026822358,
                     -0.03344538412056863,
                     -0.06456210720352829,
                     0.3943447499535977,
                     0.5358478897251189,
                     -0.8607350478880107]

# Choose whether output of the propagation is written to files
write_results_to_file = False
# Get path of current directory
current_dir = os.path.dirname(__file__)

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Vehicle settings
vehicle_mass = 4.7E3  # kg
vehicle_dry_mass = 2.25E3  # kg
constant_specific_impulse = 311.0  # s
# Fixed simulation termination settings
maximum_duration = constants.JULIAN_DAY  # s
termination_altitude = 100.0E3  # m

# Set simulation start epoch
simulation_start_epoch = 0.0  # s
# Load spice kernels
spice_interface.load_standard_kernels()
# Define settings for celestial bodies

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Initialize dictionary to save simulation output
simulation_results = dict()

bodies_to_create = ['Moon', 'Earth', 'Sun']
# Define coordinate system
global_frame_origin = 'Moon'
global_frame_orientation = 'ECLIPJ2000'
# Crate body settings
# N.B.: all the bodies added after this function is called will automatically
# be placed in the same reference frame, which is the same for the full
# system of bodies
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)
# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

###########################################################################
# CREATE VEHICLE ##########################################################
###########################################################################

# Create vehicle object and add it to the existing system of bodies
bodies.create_empty_body('Vehicle')
# Set mass of vehicle
bodies.get_body('Vehicle').set_constant_mass(vehicle_mass)

###########################################################################
# CREATE ACCELERATIONS ####################################################
###########################################################################

# Define bodies that are propagated and their central bodies of propagation
bodies_to_propagate = ['Vehicle']
central_bodies = ['Moon']
# Accelerations for the nominal model (model_test = 0)
acceleration_settings_on_vehicle = {
    'Vehicle': [get_thrust_acceleration_model_from_parameters(thrust_parameters,
                                                              bodies,
                                                              simulation_start_epoch,
                                                              constant_specific_impulse)],
    'Moon': [propagation_setup.acceleration.spherical_harmonic_gravity(2, 2)]}
# Create global accelerations dictionary
acceleration_settings = {'Vehicle': acceleration_settings_on_vehicle}

###########################################################################
# CREATE PROPAGATION SETTINGS #############################################
###########################################################################

# Retrieve termination settings
termination_settings = Util.get_termination_settings(simulation_start_epoch,
                                                     maximum_duration,
                                                     termination_altitude,
                                                     vehicle_dry_mass)
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
# Create mass rate model
mass_rate_settings_on_vehicle = {'Vehicle': [propagation_setup.mass.from_thrust()]}
# Create mass propagator settings (same for all propagations)
mass_propagator_settings = propagation_setup.propagator.mass(bodies_to_propagate,
                                                             mass_rate_settings_on_vehicle,
                                                             np.array([vehicle_mass]),
                                                             termination_settings)
# Get current propagator, and define translational state propagation settings
current_propagator = propagation_setup.propagator.TranslationalPropagatorType.cowell
# Define translational state propagation settings
translational_propagator_settings = propagation_setup.propagator.translational(central_bodies,
                                                                               acceleration_settings,
                                                                               bodies_to_propagate,
                                                                               initial_state,
                                                                               termination_settings,
                                                                               current_propagator,
                                                                               output_variables=dependent_variables_to_save)
# Note, the following line is needed to properly use the accelerations, and modify them in the Problem class
translational_propagator_settings.recreate_state_derivative_models(bodies)
# Create list of propagators, adding mass, and define full propagation settings
propagator_settings_list = [translational_propagator_settings,
                            mass_propagator_settings]
# Define full propagation settings
full_propagation_settings = propagation_setup.propagator.multitype(propagator_settings_list,
                                                                   termination_settings,
                                                                   dependent_variables_to_save)
# Create integrator settings
current_integrator_settings = Util.get_integrator_settings(0, 0, 0,
                                                           simulation_start_epoch)
# Set ranges for the decision variables ([min], [max])
decision_variable_range = ([5.0E3, 10.0, -0.1, -0.5, -0.7, -1.0, -1.3],
                           [20.0E3, 100.0, 0.1, 0.5, 0.7, 1.0, 1.3])

# Create Lunar Ascent Problem object
current_lunar_ascent_problem = LunarAscentProblem(bodies,
                                                  current_integrator_settings,
                                                  full_propagation_settings,
                                                  constant_specific_impulse,
                                                  simulation_start_epoch,
                                                  decision_variable_range)

###########################################################################
# OPTIMIZE PROBLEM ########################################################
###########################################################################

# Select algorithm from pygmo, with one generation
algo = pg.algorithm(pg.de())
# Create pygmo problem
prob = pg.problem(current_lunar_ascent_problem)
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
