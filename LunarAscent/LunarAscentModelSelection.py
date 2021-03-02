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

This function computes the dynamics of a lunar ascent vehicle, starting at zero velocity on the Moon's surface, using a
variety of model settings (see below).
 
The propagation starts near the lunar surface, with a speed relative to the Moon of 10 m/s.

The propagation is terminated as soon as one of the following conditions is met:

* Altitude > 100 km
* Altitude < 0 km
* Propagation time > 3600 s
* Vehicle mass < 2250 kg

Both the translational dynamics and mass of the vehicle are propagated, using a fixed specific impulse.
The accelerations currently considered are as follows:

0. thrust and Lunar spherical harmonic gravity up to order 2 and degree 2 (NOMINAL CASE)
1. thrust and Lunar point mass gravity
2. thrust and Lunar spherical harmonic gravity (up to order 4 and degree 4)
3. thrust, Lunar point mass gravity and Earth's point mass gravity
4. thrust and Lunar point mass gravity, with different Lunar ephemeris (generated through the unperturbed Earth-Moon
two-body problem)

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

# Set number of models to loop over
number_of_models = 5

# Set the interpolation step at which different runs are compared
output_interpolation_step = 1.0  # s

for model_test in range(5):

    bodies_to_create = ['Moon', 'Earth', 'Sun']
    # Define coordinate system
    global_frame_origin = 'Moon'
    global_frame_orientation = 'ECLIPJ2000'
    # Create body settings
    # N.B.: all the bodies added after this function is called will automatically
    # be placed in the same reference frame, which is the same for the full
    # system of bodies
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create,
        global_frame_origin,
        global_frame_orientation)

    # For case 4, the ephemeris of the Moon is generated by solving the 2-body problem of the Earth and the Moon
    # (in the other cases, the ephemeris of the Moon from SPICE takes into account all the perturbations)
    if (model_test == 4):
        effective_gravitational_parameter = spice_interface.get_body_gravitational_parameter('Moon') + \
                                            spice_interface.get_body_gravitational_parameter('Earth')
        body_settings.get('Moon').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
            'Moon', simulation_start_epoch, effective_gravitational_parameter, 'Earth', global_frame_orientation)
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

    # Here, model settings are modified
    if model_test == 1:
        acceleration_settings_on_vehicle['Moon'] = [propagation_setup.acceleration.point_mass_gravity()]
    elif model_test == 2:
        acceleration_settings_on_vehicle['Moon'] = [propagation_setup.acceleration.spherical_harmonic_gravity(4, 4)]
    elif model_test > 2:
        acceleration_settings_on_vehicle['Earth'] = [propagation_setup.acceleration.point_mass_gravity()]

    # Create global accelerations dictionary
    acceleration_settings = {'Vehicle': acceleration_settings_on_vehicle}

    ###########################################################################
    # CREATE (CONSTANT) PROPAGATION SETTINGS ##################################
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
    # Create Lunar Ascent Problem object
    current_lunar_ascent_problem = LunarAscentProblem(bodies,
                                                      current_integrator_settings,
                                                      full_propagation_settings,
                                                      constant_specific_impulse,
                                                      simulation_start_epoch)
    # Update thrust settings and evaluate fitness
    current_lunar_ascent_problem.fitness(thrust_parameters)

    # Retrieve propagated state and dependent variables
    state_history = current_lunar_ascent_problem.get_last_run_propagated_cartesian_state_history()
    dependent_variable_history = current_lunar_ascent_problem.get_last_run_dependent_variable_history()

    # Save results to a dictionary
    simulation_results[model_test] = [state_history, dependent_variable_history]

    # Set time limits to avoid numerical issues at the boundaries due to the interpolation
    propagation_times = list(state_history.keys())
    limit_times = {propagation_times[3]: propagation_times[-3]}

    # Get output path
    if model_test == 0:
        subdirectory = '/NominalCase/'
    else:
        subdirectory = '/Model_' + str(model_test) + '/'

    # Decide if output writing is required
    if write_results_to_file:
        output_path = current_dir + subdirectory
    else:
        output_path = None

    # If desired, write output to a file
    if write_results_to_file:
        save2txt(state_history, 'state_history.dat', output_path)
        save2txt(dependent_variable_history, 'dependent_variable_history.dat', output_path)
        save2txt(limit_times, 'limit_times.dat', output_path)

"""
NOTE TO STUDENTS
The first index of the dictionary simulation_results refers to the model case, while the second index can be 0 (states)
or 1 (dependent variables).
You can use this dictionary to make all the cross-comparison that you deem necessary. The code below currently compares
every case with respect to the "nominal" one.
"""
# Compare all the model settings with the nominal case
for model_test in range(1, number_of_models):
    # Get output path
    output_path = current_dir + '/Model_' + str(model_test) + '/'
    # Retrieve current state and dependent variable history
    current_state_history = simulation_results[model_test][0]
    current_dependent_variable_history = simulation_results[model_test][1]
    # Create vector of epochs to be compared (boundaries are referred to the first case)
    current_epochs = list(current_state_history.keys())
    interpolation_epochs = np.arange(current_epochs[0],
                                     current_epochs[-1],
                                     output_interpolation_step)
    # Compare state history
    state_difference_wrt_nominal = Util.compare_models(current_state_history,
                                                       simulation_results[0][0],
                                                       interpolation_epochs,
                                                       output_path,
                                                       'state_difference_wrt_nominal_case.dat')
    # Compare dependent variable history
    dependent_variable_difference_wrt_nominal = Util.compare_models(current_dependent_variable_history,
                                                                    simulation_results[0][1],
                                                                    interpolation_epochs,
                                                                    output_path,
                                                                    'dependent_variable_difference_wrt_nominal_case.dat')