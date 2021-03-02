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

The script saves the state and dependent variable history for each model settings. In addition, the differences in state
and dependent variable history with respect to the nominal case are also written to files. The output is written if
the variable write_results_to_file is true.
"""

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import os

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
write_results_to_file = True
# Get path of current directory
current_dir = os.path.dirname(__file__)

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################
for model_test in range(5):

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
    # For case 3, a different rotational model is used for the Earth
    if model_test == 3:
        body_settings.get( 'Earth' ).rotation_model_settings = environment_setup.rotation_model.simple_from_spice(
	        global_frame_orientation, 'IAU_Earth', 'IAU_Earth', simulation_start_epoch)
    # For case 4, a different atmospheric model is used
    if model_test == 4:
        density_scale_height = 7.2E3
        constant_temperature = 290
        density_at_zero_altitude = 1.225
        specific_gas_constant = 287.06
        body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.exponential(
            density_scale_height, constant_temperature, density_at_zero_altitude, specific_gas_constant)

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
    # Here different acceleration models are defined
    if model_test == 1:
        acceleration_settings_on_vehicle['Earth'][0] = propagation_setup.acceleration.point_mass_gravity()
    elif model_test == 2:
        acceleration_settings_on_vehicle['Earth'][0] = propagation_setup.acceleration.spherical_harmonic_gravity(2, 2)

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
    # Create Lunar Ascent Problem object
    current_shape_optimization_problem = ShapeOptimizationProblem(bodies,
                                                                  current_integrator_settings,
                                                                  current_propagator_settings,
                                                                  capsule_density)
    # Update thrust settings and evaluate fitness
    current_shape_optimization_problem.fitness(shape_parameters)

    # Retrieve propagated state and dependent variables
    state_history = current_shape_optimization_problem.get_last_run_propagated_cartesian_state_history()
    dependent_variable_history = current_shape_optimization_problem.get_last_run_dependent_variable_history()

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
    if (model_test == 0):
        nominal_interpolator_settings = interpolators.lagrange_interpolation(8)
        nominal_state_interpolator = interpolators.create_one_dimensional_interpolator(
            state_history, nominal_interpolator_settings)
        nominal_dependent_variable_interpolator = interpolators.create_one_dimensional_interpolator(
            dependent_variable_history, nominal_interpolator_settings)
    # Compare current model with nominal case
    else:
        state_difference = dict()
        for epoch in state_history.keys():
            state_difference[epoch] = state_history[epoch] - nominal_state_interpolator.interpolate(epoch)
        dependent_difference = dict()
        # Loop over the propagated dependent variables and use the benchmark interpolators
        for epoch in dependent_variable_history.keys():
            dependent_difference[epoch] = dependent_variable_history[
                                              epoch] - nominal_dependent_variable_interpolator.interpolate(epoch)
        # If desired, write differences to files
        if write_results_to_file:
            save2txt(state_difference, 'state_difference.dat', output_path)
            save2txt(dependent_difference, 'dependent_variable_difference.dat', output_path)