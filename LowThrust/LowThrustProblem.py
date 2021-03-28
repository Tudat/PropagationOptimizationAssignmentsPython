'''
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

This module contains the problem-specific classes and functions, which will be called by the main script where the
optimization is executed.
'''

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np

# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.simulation import propagation_setup
from tudatpy.kernel.simulation import shape_based_thrust
from tudatpy.util import result2array

# Problem-specific imports
import LowThrustUtilities as Util


###########################################################################
# USEFUL PROBLEM-SPECIFIC FUNCTIONS ########################################
###########################################################################

def compute_fitness(trajectory_parameters: list,
                    hodographic_shaping_object: tudatpy.kernel.simulation.shape_based_thrust.HodographicShaping,
                    state_history: dict = None,
                    dependent_variable_history: dict = None) -> list:
    """
    It computes the fitness values. Currently, the (single) objective is to minimize the delta v.
    This function should be improved and/or extended.

    Parameters
    ----------
    trajectory_parameters:
        List of decision variables.
    hodographic_shaping_object : tudatpy.kernel.simulation.shape_based_thrust.HodographicShaping
        Hodographic shaping object.
    state_history : dict
        State history.
    dependent_variable_history : dict
        Dependent variable history.

    Returns
    -------
    list
        List of fitness values for each objective.
    """
    # Converts state and dependent variable history to arrays
    if state_history is not None:
        states = result2array(state_history)
        dependent_variables = result2array(dependent_variable_history)
    # Computes the delta V consumed
    delta_v = hodographic_shaping_object.compute_delta_v()
    return [delta_v]


def get_trajectory_time_of_flight(trajectory_parameters: list) -> float:
    """
    Returns the time of flight in seconds.

    Parameters
    ----------
    trajectory_parameters : list of floats
        List of trajectory parameters to optimize.

    Returns
    -------
    float
        Time of flight [s].
    """
    return trajectory_parameters[1] * constants.JULIAN_DAY


def get_trajectory_initial_time(trajectory_parameters: list,
                                buffer_time: float = 0.0) -> float:
    """
    Returns the time of flight in seconds.

    Parameters
    ----------
    trajectory_parameters : list of floats
        List of trajectory parameters to optimize.
    buffer_time : float (default: 0.0)
        Delay between start of the hodographic trajectory and the start of the propagation.

    Returns
    -------
    float
        Initial time of the hodographic trajectory [s].
    """
    return trajectory_parameters[0] * constants.JULIAN_DAY + buffer_time


def get_trajectory_final_time(trajectory_parameters: list,
                              buffer_time: float = 0.0) -> float:
    """
    Returns the time of flight in seconds.

    Parameters
    ----------
    trajectory_parameters : list of floats
        List of trajectory parameters to optimize.
    buffer_time : float (default: 0.0)
        Delay between start of the hodographic trajectory and the start of the propagation.

    Returns
    -------
    float
        Final time of the hodographic trajectory [s].
    """
    # Get initial time
    initial_time = get_trajectory_initial_time(trajectory_parameters)
    return initial_time + get_trajectory_time_of_flight(trajectory_parameters) - buffer_time


def get_hodographic_trajectory(shaping_object: tudatpy.kernel.simulation.shape_based_thrust.HodographicShaping,
                               trajectory_parameters: list,
                               specific_impulse: float,
                               output_path: str = None):
    """
    It computes the analytical hodographic trajectory and saves the results to a file, if desired.

    This function analytically calculates the hodographic trajectory from the Hodographic Shaping object. It
    retrieves both the trajectory and the acceleration profile; if desired, both are saved to files as follows:

    * hodographic_trajectory.dat: Cartesian states of semi-analytical trajectory;
    * hodographic_thrust_acceleration.dat: Thrust acceleration in inertial, Cartesian, coordinates, along the
    semi-analytical trajectory.

    NOTE: The independent variable (first column) does not represent the usual time (seconds since J2000), but instead
    denotes the time since departure.

    Parameters
    ----------
    shaping_object: tudatpy.kernel.simulation.shape_based_thrust.HodographicShaping
        Hodographic shaping object.
    trajectory_parameters : list of floats
        List of trajectory parameters to be optimized.
    specific_impulse : float
        Constant specific impulse of the spacecraft.
    output_path : str (default: None)
        If and where to save the benchmark results (if None, results are NOT written).

    Returns
    -------
    none
    """
    # Set time parameters
    start_time = 0.0
    final_time = get_trajectory_time_of_flight(trajectory_parameters)
    # Set number of data points
    number_of_data_points = 10000
    # Compute step size
    step_size = (final_time - start_time) / (number_of_data_points - 1)
    # Create epochs vector
    epochs = np.linspace(start_time,
                         final_time,
                         number_of_data_points)
    # Create specific impulse lambda function
    specific_impulse_function = lambda t: specific_impulse
    # Retrieve thrust acceleration profile from shaping object
    # NOTE TO THE STUDENTS: do not uncomment
    # thrust_acceleration_profile = shaping_object.get_thrust_acceleration_profile(
    #     epochs,
    #     specific_impulse_function)
    # Retrieve trajectory from shaping object
    trajectory_shape = shaping_object.get_trajectory(epochs)
    # If desired, save results to files
    if output_path is not None:
        # NOTE TO THE STUDENTS: do not uncomment
        # save2txt(thrust_acceleration_profile,
        #          'hodographic_thrust_acceleration.dat',
        #          output_path)
        save2txt(trajectory_shape,
                 'hodographic_trajectory.dat',
                 output_path)


def get_radial_velocity_shaping_functions(trajectory_parameters: list,
                                          frequency: float,
                                          scale_factor: float,
                                          time_of_flight: float,
                                          number_of_revolutions: int) -> tuple:
    """
    Retrieves the radial velocity shaping functions (lowest and highest order in Gondelach and Noomen, 2015) and returns
    them together with the free coefficients.

    Parameters
    ----------
    trajectory_parameters : list
        List of trajectory parameters to optimize.
    frequency: float
        Frequency of the highest-order methods.
    scale_factor: float
        Scale factor of the highest-order methods.
    time_of_flight: float
        Time of flight of the trajectory.
    number_of_revolutions: int
        Number of revolutions around the Sun (currently unused).

    Returns
    -------
    tuple
        A tuple composed by two lists: the radial velocity shaping functions and their free coefficients.
    """
    # Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
    radial_velocity_shaping_functions = shape_based_thrust.recommended_radial_hodograph_functions(time_of_flight)
    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    radial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
        exponent=1.0,
        frequency=0.5 * frequency,
        scale_factor=scale_factor))
    radial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
        exponent=1.0,
        frequency=0.5 * frequency,
        scale_factor=scale_factor))
    # Set free parameters
    free_coefficients = trajectory_parameters[3:5]
    return (radial_velocity_shaping_functions,
            free_coefficients)


def get_normal_velocity_shaping_functions(trajectory_parameters: list,
                                          frequency: float,
                                          scale_factor: float,
                                          time_of_flight: float,
                                          number_of_revolutions: int) -> tuple:
    """
    Retrieves the normal velocity shaping functions (lowest and highest order in Gondelach and Noomen, 2015) and returns
    them together with the free coefficients.

    Parameters
    ----------
    trajectory_parameters : list
        List of trajectory parameters to optimize.
    frequency: float
        Frequency of the highest-order methods.
    scale_factor: float
        Scale factor of the highest-order methods.
    time_of_flight: float
        Time of flight of the trajectory.
    number_of_revolutions: int
        Number of revolutions around the Sun (currently unused).

    Returns
    -------
    tuple
        A tuple composed by two lists: the normal velocity shaping functions and their free coefficients.
    """
    # Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
    normal_velocity_shaping_functions = shape_based_thrust.recommended_normal_hodograph_functions(time_of_flight)
    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    normal_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
        exponent=1.0,
        frequency=0.5 * frequency,
        scale_factor=scale_factor))
    normal_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
        exponent=1.0,
        frequency=0.5 * frequency,
        scale_factor=scale_factor))
    # Set free parameters
    free_coefficients = trajectory_parameters[5:7]
    return (normal_velocity_shaping_functions,
            free_coefficients)


def get_axial_velocity_shaping_functions(trajectory_parameters: list,
                                         frequency: float,
                                         scale_factor: float,
                                         time_of_flight: float,
                                         number_of_revolutions: int) -> tuple:
    """
    Retrieves the axial velocity shaping functions (lowest and highest order in Gondelach and Noomen, 2015) and returns
    them together with the free coefficients.

    Parameters
    ----------
    trajectory_parameters : list
        List of trajectory parameters to optimize.
    frequency: float
        Frequency of the highest-order methods.
    scale_factor: float
        Scale factor of the highest-order methods.
    time_of_flight: float
        Time of flight of the trajectory.
    number_of_revolutions: int
        Number of revolutions around the Sun.

    Returns
    -------
    tuple
        A tuple composed by two lists: the axial velocity shaping functions and their free coefficients.
    """
    # Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
    axial_velocity_shaping_functions = shape_based_thrust.recommended_axial_hodograph_functions(
        time_of_flight,
        number_of_revolutions)
    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    exponent = 4.0
    axial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
        exponent=exponent,
        frequency=(number_of_revolutions + 0.5) * frequency,
        scale_factor=scale_factor ** exponent))
    axial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
        exponent=exponent,
        frequency=(number_of_revolutions + 0.5) * frequency,
        scale_factor=scale_factor ** exponent))
    # Set free parameters
    free_coefficients = trajectory_parameters[7:9]
    return (axial_velocity_shaping_functions,
            free_coefficients)


def create_hodographic_shaping_object(trajectory_parameters: list,
                                      bodies: tudatpy.kernel.simulation.environment_setup.SystemOfBodies) \
        -> tudatpy.kernel.simulation.shape_based_thrust.HodographicShaping:
    """
    It creates and returns the hodographic shaping object, based on the trajectory parameters.

    Parameters
    ----------
    trajectory_parameters : list
        List of trajectory parameters to be optimized.
    bodies : tudatpy.kernel.simulation.environment_setup.SystemOfBodies
        System of bodies present in the simulation.

    Returns
    -------
    hodographic_shaping_object : tudatpy.kernel.simulation.shape_based_thrust.HodographicShaping
        Hodographic shaping object.
    """
    # Time settings
    initial_time = get_trajectory_initial_time(trajectory_parameters)
    time_of_flight = get_trajectory_time_of_flight(trajectory_parameters)
    final_time = get_trajectory_final_time(trajectory_parameters)
    # Number of revolutions
    number_of_revolutions = int(trajectory_parameters[2])
    # Compute relevant frequency and scale factor for shaping functions
    frequency = 2.0 * np.pi / time_of_flight
    scale_factor = 1.0 / time_of_flight
    # Retrieve shaping functions and free parameters
    radial_velocity_shaping_functions, radial_free_coefficients = get_radial_velocity_shaping_functions(
        trajectory_parameters,
        frequency,
        scale_factor,
        time_of_flight,
        number_of_revolutions)
    normal_velocity_shaping_functions, normal_free_coefficients = get_normal_velocity_shaping_functions(
        trajectory_parameters,
        frequency,
        scale_factor,
        time_of_flight,
        number_of_revolutions)
    axial_velocity_shaping_functions, axial_free_coefficients = get_axial_velocity_shaping_functions(
        trajectory_parameters,
        frequency,
        scale_factor,
        time_of_flight,
        number_of_revolutions)
    # Retrieve boundary conditions and central body gravitational parameter
    initial_state = bodies.get_body('Earth').get_state_in_based_frame_from_ephemeris(initial_time)
    final_state = bodies.get_body('Mars').get_state_in_based_frame_from_ephemeris(final_time)
    gravitational_parameter = bodies.get_body('Sun').gravitational_parameter
    # Create and return shape-based method
    hodographic_shaping_object = shape_based_thrust.HodographicShaping(initial_state,
                                                                       final_state,
                                                                       time_of_flight,
                                                                       gravitational_parameter,
                                                                       number_of_revolutions,
                                                                       radial_velocity_shaping_functions,
                                                                       normal_velocity_shaping_functions,
                                                                       axial_velocity_shaping_functions,
                                                                       radial_free_coefficients,
                                                                       normal_free_coefficients,
                                                                       axial_free_coefficients)
    return hodographic_shaping_object


def get_hodograph_thrust_acceleration_settings(trajectory_parameters: list,
                                               bodies: tudatpy.kernel.simulation.environment_setup.SystemOfBodies,
                                               specific_impulse: float) \
        -> tudatpy.kernel.simulation.shape_based_thrust.HodographicShaping:
    """
    It extracts the acceleration settings resulting from the hodographic trajectory and returns the equivalent thrust
    acceleration settings object.

    Parameters
    ----------
    trajectory_parameters : list
        List of trajectory parameters to be optimized.
    bodies : tudatpy.kernel.simulation.environment_setup.SystemOfBodies
        System of bodies present in the simulation.
    specific_impulse : float
        Constant specific impulse of the spacecraft.

    Returns
    -------
    tudatpy.kernel.simulation.propagation_setup.acceleration.ThrustAccelerationSettings
        Thrust acceleration settings object.
    """
    # Create shaping object
    shaping_object = create_hodographic_shaping_object(trajectory_parameters,
                                                       bodies)
    # Compute offset, which is the time since J2000 (when t=0 for tudat) at which the simulation starts
    # N.B.: this is different from time_buffer, which is the delay between the start of the hodographic
    # trajectory and the beginning of the simulation
    time_offset = get_trajectory_initial_time(trajectory_parameters)
    # Create specific impulse lambda function
    specific_impulse_function = lambda t: specific_impulse
    # Return acceleration settings
    return shape_based_thrust.get_low_thrust_acceleration_settings(shaping_object,
                                                                   bodies,
                                                                   'Vehicle',
                                                                   specific_impulse_function,
                                                                   time_offset)


def get_hodograph_state_at_epoch(trajectory_parameters: list,
                                 bodies: tudatpy.kernel.simulation.environment_setup.SystemOfBodies,
                                 epoch: float) -> np.ndarray:
    """
    It retrieves the Cartesian state, expressed in the inertial frame, at a given epoch of the analytical trajectory.

    Parameters
    ----------
    trajectory_parameters : list
        List of trajectory parameters to be optimized.
    bodies : tudatpy.kernel.simulation.environment_setup.SystemOfBodies
        System of bodies present in the simulation.

    Returns
    -------
    np.ndarray
        Cartesian state in the inertial frame of the spacecraft at the given epoch.
    """
    # Create shaping object
    shaping_object = create_hodographic_shaping_object(trajectory_parameters,
                                                       bodies)
    # Define current hodograph time
    hodograph_time = epoch - get_trajectory_initial_time(trajectory_parameters)
    return shaping_object.get_state(hodograph_time)


###########################################################################
# CREATE PROBLEM CLASS ####################################################
###########################################################################

class LowThrustProblem:
    """
    Class to initialize, simulate and optimize the Low Thrust trajectory.

    The class is created specifically for this problem. This is done to provide better integration with Pagmo/Pygmo,
    where the optimization process (assignment 3) will be done. For the first two assignments, the presence of this
    class is not strictly needed, but it was chosen to create the code so that the same version could be used for all
    three assignments.

    Attributes
    ----------
    bodies
    integrator_settings
    propagator_settings
    specific_impulse
    minimum_mars_distance
    time_buffer
    perform_propagation

    Methods
    -------
    get_last_run_propagated_cartesian_state_history()
    get_last_run_propagated_state_history()
    get_last_run_dependent_variable_history()
    get_last_run_dynamics_simulator()
    get_bounds()
    get_nobj()
    fitness(trajectory_parameters)
    get_hodographic_shaping()
    """

    def __init__(self,
                 bodies: tudatpy.kernel.simulation.environment_setup.SystemOfBodies,
                 integrator_settings: tudatpy.kernel.simulation.propagation_setup.integrator.IntegratorSettings,
                 propagator_settings: tudatpy.kernel.simulation.propagation_setup.propagator.MultiTypePropagatorSettings,
                 specific_impulse: float,
                 minimum_mars_distance: float,
                 time_buffer: float,
                 decision_variable_range: tuple,
                 perform_propagation: bool = True):
        """
        Constructor for the LowThrustProblem class.

        Parameters
        ----------
        bodies : tudatpy.kernel.simulation.environment_setup.SystemOfBodies,
            System of bodies present in the simulation.
        integrator_settings : tudatpy.kernel.simulation.propagation_setup.integrator.IntegratorSettings
            Integrator settings to be provided to the dynamics simulator.
        propagator_settings : tudatpy.kernel.simulation.propagation_setup.propagator.MultiTypePropagatorSettings
            Propagator settings object.
        specific_impulse : float
            Constant specific impulse of the vehicle.
        minimum_mars_distance : float
            Minimum distance from Mars at which the propagation stops.
        time_buffer : float
            Time interval between the simulation start epoch and the beginning of the hodographic trajectory.
        decision_variable_range : tuple
            Boundary values (min, max) for the decision variables.
        perform_propagation : bool (default: True)
            If true, the propagation is performed.

        Returns
        -------
        none
        """
        # Copy arguments as attributes
        self.bodies_function = lambda: bodies
        self.integrator_settings_function = lambda: integrator_settings
        self.propagator_settings_function = lambda: propagator_settings
        self.specific_impulse = specific_impulse
        self.minimum_mars_distance = minimum_mars_distance
        self.time_buffer = time_buffer
        self.decision_variable_range = decision_variable_range
        self.perform_propagation = perform_propagation
        # Extract translational state propagator settings from the full propagator settings
        if perform_propagation:
            self.translational_state_propagator_settings_function = lambda: propagator_settings.single_type_settings(
                propagation_setup.translational_type)

    def get_bounds(self):
        """
        Returns the boundaries of the decision variables.
        """

        return self.decision_variable_range

    def get_nobj(self):
        """
        Returns the number of objectives (currently: single-objective).
        """

        return 1

    def get_last_run_propagated_cartesian_state_history(self) -> dict:
        """
        Returns the full history of the propagated state, converted to Cartesian states

        Parameters
        ----------
        none

        Returns
        -------
        dict
        """
        return self.dynamics_simulator_function().get_equations_of_motion_numerical_solution()

    def get_last_run_propagated_state_history(self) -> dict:
        """
        Returns the full history of the propagated state, not converted to Cartesian state
        (i.e. in the actual formulation that was used during the numerical integration).

        Parameters
        ----------
        none

        Returns
        -------
        dict
        """
        return self.dynamics_simulator_function().get_equations_of_motion_numerical_solution_raw()

    def get_last_run_dependent_variable_history(self) -> dict:
        """
        Returns the full history of the dependent variables.

        Parameters
        ----------
        none

        Returns
        -------
        dict
        """
        return self.dynamics_simulator_function().get_dependent_variable_history()

    def get_last_run_dynamics_simulator(self) -> tudatpy.kernel.simulation.propagation_setup.SingleArcDynamicsSimulator:
        """
        Returns the dynamics simulator object.

        Parameters
        ----------
        none

        Returns
        -------
        tudatpy.kernel.simulation.propagation_setup.SingleArcDynamicsSimulator
        """
        return self.dynamics_simulator_function

    def fitness(self,
                trajectory_parameters) -> float:
        """
        Propagate the trajectory using the hodographic method with the parameters given as argument.

        This function uses the trajectory parameters to create a new hodographic shaping object, from which a new
        thrust acceleration profile is extracted. Subsequently, the trajectory is propagated numerically, if desired.
        The fitness, currently set to zero, can be computed here: it will be used during the optimization process.

        Parameters
        ----------
        trajectory_parameters : list of floats
            List of trajectory parameters to optimize.

        Returns
        -------
        fitness : float
            Fitness value (for optimization, see assignment 3).
        """
        # Create hodographic shaping object
        bodies = self.bodies_function()
        hodographic_shaping = create_hodographic_shaping_object(trajectory_parameters,
                                                                bodies)
        self.hodographic_shaping_function = lambda: hodographic_shaping
        # Propagate trajectory only if required
        if self.perform_propagation:
            integrator_settings = self.integrator_settings_function()
            translational_state_propagator_settings = self.translational_state_propagator_settings_function()
            propagator_settings = self.propagator_settings_function()

            initial_propagation_time = get_trajectory_initial_time(trajectory_parameters,
                                                                   self.time_buffer)
            # Reset initial time
            integrator_settings.initial_time = initial_propagation_time
            # Retrieve the accelerations from the translational state propagator
            acceleration_settings = translational_state_propagator_settings.acceleration_settings
            # Clear the existing thrust acceleration
            acceleration_settings['Vehicle']['Vehicle'].clear()
            # Create specific impulse lambda function
            specific_impulse_function = lambda t: self.specific_impulse
            # Compute offset, which is the time since J2000 (when t=0 for tudat) at which the simulation starts
            # N.B.: this is different from time_buffer, which is the delay between the start of the hodographic
            # trajectory and the beginning of the simulation
            time_offset = get_trajectory_initial_time(trajectory_parameters)
            # Retrieve new thrust settings
            new_thrust_settings = shape_based_thrust.get_low_thrust_acceleration_settings(
                hodographic_shaping,
                bodies,
                'Vehicle',
                specific_impulse_function,
                time_offset)
            # Set new acceleration settings
            acceleration_settings['Vehicle']['Vehicle'].append(new_thrust_settings)
            # Update translational propagator settings: accelerations
            translational_state_propagator_settings.reset_and_recreate_acceleration_models(acceleration_settings,
                                                                                           bodies)
            # Retrieve initial state
            new_initial_state = get_hodograph_state_at_epoch(trajectory_parameters,
                                                             bodies,
                                                             initial_propagation_time)
            # Update translational propagator settings: initial state
            translational_state_propagator_settings.reset_initial_states(new_initial_state)
            # Update full propagator settings
            propagator_settings.recreate_state_derivative_models(bodies)
            # Reset full initial state
            new_full_initial_state = propagation_setup.propagator.combine_initial_states(
                propagator_settings.propagator_settings_per_type)
            propagator_settings.reset_initial_states(new_full_initial_state)
            # Get the termination settings
            propagator_settings.termination_settings = Util.get_termination_settings(trajectory_parameters,
                                                                                     self.minimum_mars_distance,
                                                                                     self.time_buffer)
            # Create simulation object and propagate dynamics
            dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(bodies,
                                                                              integrator_settings,
                                                                              propagator_settings,
                                                                              print_dependent_variable_data=False)

            self.dynamics_simulator_function = lambda: dynamics_simulator
        # Compute fitness, checking if the trajectory has been numerically propagated or not
        if hasattr(self, 'dynamics_simulator_function'):
            fitness_list = compute_fitness(trajectory_parameters,
                                           hodographic_shaping,
                                           self.get_last_run_propagated_cartesian_state_history(),
                                           self.get_last_run_dependent_variable_history())
        else:
            fitness_list = compute_fitness(trajectory_parameters,
                                           hodographic_shaping)
        return fitness_list

    def get_hodographic_shaping(self):
        """
        Returns the hodographic shaping object.

        Parameters
        ----------
        none

        Returns
        -------
        tudatpy.kernel.simulation.shape_based_thrust.HodographicShaping
        """
        return self.hodographic_shaping_function()
