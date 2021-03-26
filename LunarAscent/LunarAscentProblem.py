'''
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
from tudatpy.kernel import constants
from tudatpy.kernel.simulation import propagation_setup
from tudatpy.kernel.astro import frames
from tudatpy.kernel.math import interpolators

###########################################################################
# CREATE PROBLEM CLASS ####################################################
###########################################################################

class LunarAscentProblem:
    """
    Class to initialize, simulate and optimize the Lunar Ascent.

    The class is created specifically for this problem. This is done to provide better integration with Pagmo/Pygmo,
    where the optimization process (assignment 3) will be done. For the first two assignments, the presence of this
    class is not strictly needed, but it was chosen to create the code so that the same version could be used for all
    three assignments.

    Attributes
    ----------
    bodies
    integrator_settings
    propagator_settings
    constant_specific_impulse
    simulation_start_epoch
    dynamics_simulator

    Methods
    -------
    get_last_run_propagated_state_history()
    get_last_run_dependent_variable_history()
    get_last_run_dynamics_simulator()
    fitness(thrust_parameters)
    """

    def __init__(self,
                 bodies: tudatpy.kernel.simulation.environment_setup.SystemOfBodies,
                 integrator_settings: tudatpy.kernel.simulation.propagation_setup.integrator.IntegratorSettings,
                 propagator_settings: tudatpy.kernel.simulation.propagation_setup.propagator.MultiTypePropagatorSettings,
                 constant_specific_impulse: float,
                 simulation_start_epoch: float,
                 decision_variable_range):
        """
        Constructor for the LunarAscentProblem class.

        Parameters
        ----------
        bodies : tudatpy.kernel.simulation.environment_setup.SystemOfBodies
            System of bodies present in the simulation.
        integrator_settings : tudatpy.kernel.simulation.propagation_setup.integrator.IntegratorSettings
            Integrator settings to be provided to the dynamics simulator.
        propagator_settings : tudatpy.kernel.simulation.propagation_setup.propagator.MultiTypePropagatorSettings
            Propagator settings object.
        constant_specific_impulse : float
            Specific impulse of the vehicle that is kept constant during the propagation.
        simulation_start_epoch : float
            Epoch when the simulation begins [s].

        Returns
        -------
        none
        """
        # Set attributes
        self.bodies_function = lambda : bodies
        self.integrator_settings_function = lambda :integrator_settings
        self.propagator_settings_function = lambda :propagator_settings
        self.constant_specific_impulse = constant_specific_impulse
        self.simulation_start_epoch = simulation_start_epoch
        self.decision_variable_range = decision_variable_range

        # Extract translational state propagator settings from the full propagator settings
        translational_type = propagation_setup.StateType.translational_type
        self.translational_state_propagator_settings_function = lambda :propagator_settings.single_type_settings(translational_type)

    def get_bounds(self):

        return self.decision_variable_range

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
        return self.dynamics_simulator_function( ).get_equations_of_motion_numerical_solution()

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
        return self.dynamics_simulator_function( ).get_equations_of_motion_numerical_solution_raw()

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
        return self.dynamics_simulator_function( ).get_dependent_variable_history()

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
        return self.dynamics_simulator_function( )

    def fitness(self,
                thrust_parameters: list) -> float:
        """
        Propagate the trajectory with the thrust parameters given as argument.

        This function uses the thrust parameters to set a new acceleration model, subsequently propagating the
        trajectory. The fitness, currently set to zero, can be computed here: it will be used during the optimization
        process.

        Parameters
        ----------
        thrust_parameters : list of floats
            List of thrust parameters.

        Returns
        -------
        fitness : float
            Fitness value (for optimization, see assignment 3).
        """

        bodies = self.bodies_function()
        integrator_settings = self.integrator_settings_function()
        translational_state_propagator_settings = self.translational_state_propagator_settings_function()
        propagator_settings = self.propagator_settings_function()

        # Retrieve the accelerations from the translational state propagator
        acceleration_settings = translational_state_propagator_settings.acceleration_settings
        # Clear the existing thrust acceleration
        acceleration_settings['Vehicle']['Vehicle'].clear()
        # Get the new thrust settings
        new_thrust_settings = get_thrust_acceleration_model_from_parameters(thrust_parameters,
                                                                            bodies,
                                                                            self.simulation_start_epoch,
                                                                            self.constant_specific_impulse)
        # Set new acceleration settings
        acceleration_settings['Vehicle']['Vehicle'].append(new_thrust_settings)
        # Update translational propagator settings
        translational_state_propagator_settings.reset_and_recreate_acceleration_models(acceleration_settings,
                                                                                            bodies)
        # Update full propagator settings
        propagator_settings.recreate_state_derivative_models(bodies)
        # Create simulation object and propagate dynamics
        dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(bodies,
                                                                               integrator_settings,
                                                                               propagator_settings,
                                                                               print_dependent_variable_data = False)
        self.dynamics_simulator_function = lambda: dynamics_simulator

        # For the first two assignments, no computation of fitness is needed
        fitness = 0.0
        return [fitness]


###########################################################################
# CREATE GUIDANCE CLASS ###################################################
###########################################################################

class LunarAscentThrustGuidance:
    """
    Class that defines and updates the thrust guidance of the Lunar Ascent problem at each time step.

    Attributes
    ----------
    vehicle_body
    initial_time
    parameter_vector

    Methods
    -------
    get_current_thrust_magnitude(time)
    get_current_thrust_direction(time)
    """

    def __init__(self,
                 vehicle_body: str,
                 initial_time: float,
                 parameter_vector: list):
        """
        Constructor of the LunarAscentThrustGuidance class.

        Parameters
        ----------
        vehicle_body : str
            Name of the vehicle to apply the thrust guidance to.
        initial_time: float
            Initial time of the simulation [s].
        parameter_vector : list
            List of thrust parameters to retrieve the thrust guidance from.

        Returns
        -------
        none
        """
        # Set arguments as attributes
        self.vehicle_body = vehicle_body
        self.initial_time = initial_time
        self.parameter_vector = parameter_vector
        self.thrust_magnitude = parameter_vector[0]
        self.time_interval = parameter_vector[1]
        # Prepare dictionary for thrust angles
        self.thrust_angle_dict = {}
        # Initialize time
        current_time = initial_time
        # Loop over nodes
        for i in range(len(parameter_vector) - 2):
            # Store time as key, thrust angle as value
            self.thrust_angle_dict[current_time] = parameter_vector[i + 2]
            # Increase time
            current_time += self.time_interval
        # Create interpolator settings
        interpolator_settings = interpolators.linear_interpolation(
            boundary_interpolation=interpolators.use_boundary_value)
        # Create the interpolator between nodes and set it as attribute
        self.thrust_angle_interpolator = interpolators.create_one_dimensional_interpolator(self.thrust_angle_dict,
                                                                                           interpolator_settings)

    def get_current_thrust_direction(self,
                                     time: float) -> np.array:
        """
        Retrieves the direction of the thrust in the inertial frame.

        This function is needed to get the thrust direction at each time step of the propagation, based on the thrust
        parameters provided.

        Parameters
        ----------
        time : float
            Current time.

        Returns
        -------
        thrust_inertial_frame : np.array
            Thrust direction expressed in the inertial frame.
        """
        # Interpolate with time
        angle = self.thrust_angle_interpolator.interpolate(time)
        # Set thrust in vertical frame and transpose it
        thrust_direction_vertical_frame = np.array([[0, np.sin(angle), - np.cos(angle)]]).T
        # Update flight conditions (this is needed to let tudat know to update all variables)
        self.vehicle_body.flight_conditions.update_conditions(time)
        # Get aerodynamic angle calculator
        aerodynamic_angle_calculator = self.vehicle_body.flight_conditions.get_aerodynamic_angle_calculator()
        # Retrieve rotation matrix from vertical to inertial frame from the aerodynamic angle calculator
        vertical_to_inertial_frame = aerodynamic_angle_calculator.get_rotation_matrix_between_frames(
            frames.AerodynamicsReferenceFrames.vertical_frame,
            frames.AerodynamicsReferenceFrames.inertial_frame)
        # Compute the thrust in the inertial frame
        thrust_inertial_frame = np.dot(vertical_to_inertial_frame,
                                       thrust_direction_vertical_frame)
        return thrust_inertial_frame

    def get_current_thrust_magnitude(self,
                                     time: float) -> float:
        """
        Retrieves the magnitude of the thrust.

        This function is needed to get the thrust magnitude at each time step of the propagation, based on the thrust
        parameters provided. Currently, the thrust magnitude is constant.

        Parameters
        ----------
        time : float
            Current time.

        Returns
        -------
        float
            Thrust magnitude.
        """
        return self.thrust_magnitude

###########################################################################
# CREATE OTHER PROBLEM-SPECIFIC FUNCTIONS #################################
###########################################################################

def get_thrust_acceleration_model_from_parameters(thrust_parameters: list,
                                                  bodies: tudatpy.kernel.simulation.environment_setup.SystemOfBodies,
                                                  initial_time: float,
                                                  specific_impulse: float) -> \
        tudatpy.kernel.simulation.propagation_setup.acceleration.ThrustAccelerationSettings:
    """
    Creates the thrust acceleration models from the LunarAscentThrustGuidance class and sets it in the propagator.

    Parameters
    ----------
    thrust_parameters : list of floats
        List of thrust parameters.
    bodies : tudatpy.kernel.simulation.environment_setup.SystemOfBodies
        System of bodies present in the simulation.
    initial_time : float
        The start time of the simulation in seconds.
    specific_impulse : float
        Specific impulse of the vehicle.

    Returns
    -------
    tudatpy.kernel.simulation.propagation_setup.acceleration.ThrustAccelerationSettings
        Thrust acceleration settings object.
    """
    # Create Thrust Guidance object
    thrust_guidance = LunarAscentThrustGuidance(bodies.get_body('Vehicle'),
                                                initial_time,
                                                thrust_parameters)
    # Retrieves thrust functions
    thrust_direction_function = thrust_guidance.get_current_thrust_direction
    thrust_magnitude_function = thrust_guidance.get_current_thrust_magnitude
    # Set thrust functions in the acceleration model
    thrust_direction_settings = propagation_setup.acceleration.custom_thrust_direction(thrust_direction_function)
    thrust_magnitude_settings = propagation_setup.acceleration.custom_thrust_magnitude(thrust_magnitude_function,
                                                                                       lambda time: specific_impulse)
    # Create and return thrust acceleration settings
    return propagation_setup.acceleration.ThrustAccelerationSettings(thrust_direction_settings,
                                                                     thrust_magnitude_settings)