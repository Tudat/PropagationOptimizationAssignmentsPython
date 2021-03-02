"""
Copyright (c) 2010-2021, Delft University of Technology
All rights reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

AE4866 Propagation and Optimization in Astrodynamics
Low Thrust
First name: ***COMPLETE HERE***
Last name: ***COMPLETE HERE***
Student number: ***COMPLETE HERE***

This module computes the dynamics of an interplanetary low-thrust trajectory, using a thrust profile determined from
a Hodographic shaping method (see Gondelach and Noomen, 2015). This file propagates the dynamics using a variety of
model settings (see below). For each run, the differences in state and dependent variable history with respect to a
nominal case are computed.

The low-thrust trajectory computed by the shape-based method starts at the Earth's center of mass, and terminates at
Mars's center of mass.

The vehicle starts on the Hodographic low-thrust trajectory, 30 days (defined by the time_buffer variable) after it
'departs' the Earth's center of mass.

The propagation is terminated as soon as one of the following conditions is met (ses
get_propagation_termination_settings() function):

* Distance to Mars < 50000 km
* Propagation time > Time-of-flight of hodographic trajectory

This propagation assumes only point mass gravity by the Sun and thrust acceleration of the vehicle
(see block 'CREATE ACCELERATIONS'). Both the translational dynamics and mass of the vehicle are propagated,
using a fixed specific impulse. The model settings used are as follows:

0. Thrust and point mass gravity from Sun, Earth, Mars (NOMINAL CASE)
1. Thrust and point mass gravity from Sun, Earth
2. Thrust and point mass gravity from Sun, Mars
3. Thrust and point mass gravity from Sun, Earth, Mars, Jupiter
4. Thrust and point mass gravity from Sun, Earth, Mars, Jupiter with different Jovian ephemeris (generated through the
unperturbed Sun-Jupiter two-body problem)

The trajectory of the capsule is determined by its departure and arrival time (which define the initial and final states)
as well as the free parameters of the shaping method. The free parameters of the shaping method defined here are the same
as for the 'higher-order solution' in Section V.A of Gondelach and Noomen (2015). The free parameters define the amplitude
of specific types of velocity shaping functions. The low-thrust hodographic trajectory is parameterized by the values of
the vector trajectory_parameters.

The entries of the vector 'trajectoryParameters' contains the following:
* Entry 0: Departure time (from Earth's center-of-mass) in Julian days since J2000
* Entry 1: Time-of-flight from Earth's center-of-mass to Mars' center-of-mass, in Julian days
* Entry 2: Number of revolutions
* Entry 3,4: Free parameters for radial shaping functions
* Entry 5,6: Free parameters for normal shaping functions
* Entry 7,8: Free parameters for axial shaping functions

The function generating the integrator settings is unchanged with respect to assignment 1. Currently, the integrator
and integrator settings corresponding to the indices 0 and 0 are used, in combination with a Cowell propagator. Use
the combination of integrator and propagator settings that you deem the most suitable, based on the results of
assignment 1.

The script saves the state and dependent variable history for each model settings. In addition, the differences in state
and dependent variable history with respect to the nominal case are also written to files.
"""

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np
import os

# Tudatpy imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.simulation import environment_setup
from tudatpy.kernel.simulation import propagation_setup
from tudatpy.kernel.math import interpolators

# Problem-specific imports
from LowThrustProblem import LowThrustProblem, get_trajectory_initial_time, get_hodographic_trajectory
from LowThrustProblem import get_hodograph_thrust_acceleration_settings, get_hodograph_state_at_epoch
from LowThrustProblem import create_hodographic_shaping_object
import LowThrustUtilities as Util

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()
# NOTE TO STUDENTS: INPUT YOUR PARAMETER SET HERE, FROM THE INPUT FILES
# ON BRIGHTSPACE, FOR YOUR SPECIFIC STUDENT NUMBER.
# CONSIDER ALL PARAMETERS FROM YOUR SPECIFIC INPUT NUMBER, EVEN IF THE
# SIZE OF THE NEW LIST trajectory_parameters WILL BE DIFFERENT FROM THE
# SIZE OF THE ONE REPORTED BELOW. THIS IS ALREADY ACCOUNTED FOR IN THE CODE.
trajectory_parameters = [570727221.2273525 / constants.JULIAN_DAY,
                         37073942.58665284 / constants.JULIAN_DAY,
                         0,
                         2471.19649906354,
                         4207.587982407276,
                         -5594.040587888714,
                         8748.139268525232,
                         -3449.838496679572]
# Choose whether output of the propagation is written to files
write_results_to_file = False
# Get path of current directory
current_dir = os.path.dirname(__file__)

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Vehicle settings
vehicle_mass = 4.0E3
specific_impulse = 3000.0
# Fixed parameters
minimum_mars_distance = 5.0E7
# Time since 'departure from Earth CoM' at which propagation starts (and similar
# for arrival time)
time_buffer = 30.0 * constants.JULIAN_DAY

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

for model_test in range(5):
    # Define settings for celestial bodies
    bodies_to_create = ['Earth',
                        'Mars',
                        'Sun']
    if model_test > 2:
        bodies_to_create.append('Jupiter')
    # Define coordinate system
    global_frame_origin = 'SSB'
    global_frame_orientation = 'ECLIPJ2000'
    # Create body settings
    # N.B.: all the bodies added after this function is called will automatically
    # be placed in the same reference frame, which is the same for the full
    # system of bodies
    body_settings = environment_setup.get_default_body_settings(bodies_to_create,
                                                                global_frame_origin,
                                                                global_frame_orientation)
    # For case 4, the ephemeris of Jupiter is generated by solving the 2-body problem of the Sun and Jupiter
    # (in the other cases, the ephemeris of Jupiter from SPICE take into account all the perturbations)
    if (model_test == 4):
        effective_gravitational_parameter = spice_interface.get_body_gravitational_parameter('Sun') + \
                                            spice_interface.get_body_gravitational_parameter('Jupiter')
        body_settings.get('Jupiter').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
            'Jupiter', initial_propagation_time, effective_gravitational_parameter, 'Sun', global_frame_orientation)
    # Create bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    ###########################################################################
    # CREATE VEHICLE ##########################################################
    ###########################################################################

    bodies.create_empty_body('Vehicle')
    bodies.get_body('Vehicle').set_constant_mass(vehicle_mass)

    ###########################################################################
    # CREATE ACCELERATIONS ####################################################
    ###########################################################################

    # Define bodies that are propagated and their central bodies of propagation
    bodies_to_propagate = ['Vehicle']
    central_bodies = ['Sun']
    # Retrieve thrust acceleration
    thrust_settings = get_hodograph_thrust_acceleration_settings(trajectory_parameters,
                                                                 bodies,
                                                                 specific_impulse)
    # Define accelerations acting on capsule for the nominal case (model_test = 0)
    acceleration_settings_on_vehicle = {
        'Sun': [propagation_setup.acceleration.point_mass_gravity()],
        'Vehicle': [thrust_settings],
        'Mars': [propagation_setup.acceleration.point_mass_gravity()],
        'Earth': [propagation_setup.acceleration.point_mass_gravity()]
    }
    # Here model settings are modified
    if model_test == 1:
        del acceleration_settings_on_vehicle['Mars']
    elif model_test == 2:
        del acceleration_settings_on_vehicle['Earth']
    elif model_test > 2:
        acceleration_settings_on_vehicle['Jupiter'] = [propagation_setup.acceleration.point_mass_gravity()]
    # Create global accelerations dictionary
    acceleration_settings = {'Vehicle': acceleration_settings_on_vehicle}

    ###########################################################################
    # CREATE (CONSTANT) PROPAGATION SETTINGS ##################################
    ###########################################################################

    # Retrieve initial time
    initial_propagation_time = get_trajectory_initial_time(trajectory_parameters,
                                                           time_buffer)
    # Retrieve termination settings
    termination_settings = Util.get_termination_settings(trajectory_parameters,
                                                         minimum_mars_distance,
                                                         time_buffer)
    # Retrieve dependent variables to save
    dependent_variables_to_save = Util.get_dependent_variable_save_settings()
    # Check whether there is any
    if not dependent_variables_to_save:
        are_dependent_variables_to_save = False
    else:
        are_dependent_variables_to_save = True
    # Retrieve initial state
    initial_state = get_hodograph_state_at_epoch(trajectory_parameters,
                                                 bodies,
                                                 initial_propagation_time)
    # Create mass rate model
    mass_rate_settings_on_vehicle = {'Vehicle': [propagation_setup.mass.from_thrust()]}
    # Create mass propagator settings (same for all propagations)
    mass_propagator_settings = propagation_setup.propagator.mass(bodies_to_propagate,
                                                                 mass_rate_settings_on_vehicle,
                                                                 np.array([vehicle_mass]),
                                                                 termination_settings)

    ###########################################################################
    # WRITE RESULTS FOR SEMI-ANALYTICAL METHOD ################################
    ###########################################################################

    # Create problem without propagating
    hodographic_shaping_object = create_hodographic_shaping_object(trajectory_parameters,
                                                                   bodies)

    # Prepares output path
    if write_results_to_file:
        output_path = current_dir + '/SimulationOutput/HodographicSemiAnalytical/'
    else:
        output_path = None
    # Retrieves analytical results and write them to a file
    get_hodographic_trajectory(hodographic_shaping_object,
                               trajectory_parameters,
                               specific_impulse,
                               output_path)

    ###########################################################################
    # PROPAGATE TRAJECTORY ####################################################
    ###########################################################################
    # Define propagator
    current_propagator = propagation_setup.propagator.cowell
    # Define translational state propagator settings
    translational_state_propagator_settings = propagation_setup.propagator.translational(central_bodies,
                                                                                         acceleration_settings,
                                                                                         bodies_to_propagate,
                                                                                         initial_state,
                                                                                         termination_settings,
                                                                                         current_propagator,
                                                                                         dependent_variables_to_save)
    # Note, the following line is needed to properly use the accelerations, and modify them in the Problem class
    translational_state_propagator_settings.recreate_state_derivative_models(bodies)
    # Create list of propagators, adding mass, and define full propagation settings
    propagator_settings_list = [translational_state_propagator_settings,
                                mass_propagator_settings]
    full_propagation_settings = propagation_setup.propagator.multitype(propagator_settings_list,
                                                                       termination_settings,
                                                                       dependent_variables_to_save)
    # Create integrator settings
    current_integrator_settings = Util.get_integrator_settings(0,
                                                               0,
                                                               0,
                                                               initial_propagation_time)
    # Create Lunar Ascent Problem object
    current_low_thrust_problem = LowThrustProblem(bodies,
                                                  current_integrator_settings,
                                                  full_propagation_settings,
                                                  specific_impulse,
                                                  minimum_mars_distance,
                                                  time_buffer,
                                                  True)
    # Update thrust settings and evaluate fitness
    current_low_thrust_problem.fitness(trajectory_parameters)

    ###########################################################################
    # GET SIMULATION OUTPUT ###################################################
    ###########################################################################
    # Retrieve propagated state and dependent variables
    # NOTE TO STUDENTS: the following retrieve the propagated states, converted to Cartesian states
    state_history = current_low_thrust_problem.get_last_run_propagated_cartesian_state_history()
    dependent_variable_history = current_low_thrust_problem.get_last_run_dependent_variable_history()

    # Get output path
    if model_test == 0:
        subdirectory = '/NominalCase/'
    else:
        subdirectory = '/Model_' + str(model_test) + '/'
    output_path = current_dir + subdirectory

    # If desired, write output to a file
    if write_results_to_file:
        save2txt(state_history, 'state_history.dat', output_path)
        save2txt(dependent_variable_history, 'dependent_variable_history.dat', output_path)

    # Create interpolator to compare the nominal case with other settings
    if model_test == 0:
        nominal_interpolator_settings = interpolators.lagrange_interpolation(8)
        nominal_state_interpolator = interpolators.create_one_dimensional_interpolator(
            state_history, nominal_interpolator_settings)
        nominal_dependent_variable_interpolator = interpolators.create_one_dimensional_interpolator(
            dependent_variable_history, nominal_interpolator_settings)
    else:
        # Compare current model with nominal case
        state_difference = dict()
        for epoch in state_history.keys():
            state_difference[epoch] = state_history[epoch] - nominal_state_interpolator.interpolate(epoch)
        dependent_difference = dict()
        # Loop over the propagated dependent variables and use the benchmark interpolators
        for epoch in dependent_variable_history.keys():
            dependent_difference[epoch] = dependent_variable_history[epoch] - \
                                          nominal_dependent_variable_interpolator.interpolate(epoch)
        # If desired, save differences to files
        if write_results_to_file:
            save2txt(state_difference, 'state_difference.dat', output_path)
            save2txt(dependent_difference, 'dependent_variable_difference.dat', output_path)