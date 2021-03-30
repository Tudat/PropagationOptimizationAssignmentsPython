'''
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
from tudatpy.kernel.simulation import environment_setup
from tudatpy.kernel.simulation import propagation_setup
from tudatpy.kernel.astro import aerodynamics
from tudatpy.kernel.math import geometry

###########################################################################
# USEFUL PROBLEM-SPECIFIC FUNCTION DEFINITIONS ############################
###########################################################################


def get_capsule_coefficient_interface(capsule_shape: tudatpy.kernel.math.geometry.Capsule) \
        -> tudatpy.kernel.astro.aerodynamics.HypersonicLocalInclinationAnalysis:
    """
    Function that creates an aerodynamic database for a capsule, based on a set of shape parameters.

    The Capsule shape consists of four separate geometrical components: a sphere segment for the nose, a torus segment
    for the shoulder/edge, a conical frustum for the rear body, and a sphere segment for the rear cap (see Dirkx and
    Mooij, 2016). The code used in this function discretizes these surfaces into a structured mesh of quadrilateral
    panels. The parameters number_of_points and number_of_lines define the number of discretization points (for each
    part) in both independent directions (lengthwise and circumferential). The list selectedMethods defines the type of
    aerodynamic analysis method that is used.

    Parameters
    ----------
    capsule_shape : tudatpy.kernel.math.geometry.Capsule
        Object that defines the shape of the vehicle.

    Returns
    -------
    hypersonic_local_inclination_analysis : tudatpy.kernel.astro.aerodynamics.HypersonicLocalInclinationAnalysis
        Database created through the local inclination analysis method.
    """

    # Define settings for surface discretization of the capsule
    number_of_lines = [31, 31, 31, 11]
    number_of_points = [31, 31, 31, 11]
    # Set side of the vehicle (DO NOT CHANGE THESE: setting to true will turn parts of the vehicle 'inside out')
    invert_order = [0, 0, 0, 0]

    # Define moment reference point. NOTE: This value is chosen somewhat arbitrarily, and will only impact the
    # results when you consider any aspects of moment coefficients
    moment_reference = np.array([-0.6624, 0.0, 0.1369])

    # Define independent variable values
    independent_variable_data_points = []
    # Mach
    mach_points = aerodynamics.get_default_local_inclination_mach_points()
    independent_variable_data_points.append(mach_points)
    # Angle of attack
    angle_of_attack_points = np.linspace(np.deg2rad(-40),np.deg2rad(40),17)
    independent_variable_data_points.append(angle_of_attack_points)
    # Angle of sideslip
    angle_of_sideslip_points = aerodynamics.get_default_local_inclination_sideslip_angle_points()
    independent_variable_data_points.append(angle_of_sideslip_points)

    # Define local inclination method to use (index 0=Newtonian flow)
    selected_methods = [[0, 0, 0, 0], [0, 0, 0, 0]]

    # Get the capsule middle radius
    capsule_middle_radius = capsule_shape.middle_radius
    # Calculate reference area
    reference_area = np.pi * capsule_middle_radius ** 2

    # Create aerodynamic database
    hypersonic_local_inclination_analysis = aerodynamics.HypersonicLocalInclinationAnalysis(
        independent_variable_data_points,
        capsule_shape,
        number_of_lines,
        number_of_points,
        invert_order,
        selected_methods,
        reference_area,
        capsule_middle_radius,
        moment_reference)
    return hypersonic_local_inclination_analysis


def set_capsule_shape_parameters(shape_parameters: list,
                                 bodies: tudatpy.kernel.simulation.environment_setup.SystemOfBodies,
                                 capsule_density: float):
    """
    It computes and creates the properties of the capsule (shape, mass, aerodynamic coefficient interface...).

    Parameters
    ----------
    shape_parameters : list of floats
        List of shape parameters to be optimized.
    bodies : tudatpy.kernel.simulation.environment_setup.SystemOfBodies
        System of bodies present in the simulation.
    capsule_density : float
        Constant density of the vehicle.

    Returns
    -------
    none
    """
    # Compute shape constraint
    length_limit = shape_parameters[1] - shape_parameters[4] * (1 - np.cos(shape_parameters[3]))
    length_limit /= np.tan(- shape_parameters[3])
    # Add safety factor
    length_limit -= 0.01
    # Apply constraint
    if shape_parameters[2] >= length_limit:
        shape_parameters[2] = length_limit

    # Create capsule
    new_capsule = geometry.Capsule(*shape_parameters[0:5])
    # Compute new body mass
    new_capsule_mass = capsule_density * new_capsule.volume
    # Set capsule mass
    bodies.get_body('Capsule').set_constant_mass(new_capsule_mass)
    # Create aerodynamic interface from shape parameters (this calls the local inclination analysis)
    new_aerodynamic_coefficient_interface = get_capsule_coefficient_interface(new_capsule)
    # Update the Capsule's aerodynamic coefficient interface
    bodies.get_body('Capsule').set_aerodynamic_coefficient_interface(new_aerodynamic_coefficient_interface)

    return new_aerodynamic_coefficient_interface


# NOTE TO STUDENTS: if and when making modifications to the capsule shape, do include them in this function and not in
# the main code.
def add_capsule_to_body_system(bodies: tudatpy.kernel.simulation.environment_setup.SystemOfBodies,
                               shape_parameters: list,
                               capsule_density: float):
    """
    It creates the capsule body object and adds it to the body system, setting its shape based on the shape parameters
    provided.

    Parameters
    ----------
    bodies : tudatpy.kernel.simulation.environment_setup.SystemOfBodies
        System of bodies present in the simulation.
    shape_parameters : list of floats
        List of shape parameters to be optimized.
    capsule_density : float
        Constant density of the vehicle.

    Returns
    -------
    none
    """
    # Create new vehicle object and add it to the existing system of bodies
    bodies.create_empty_body('Capsule')
    # Update the capsule shape parameters
    return set_capsule_shape_parameters(shape_parameters,
                                 bodies,
                                 capsule_density)


###########################################################################
# CREATE GUIDANCE CLASS ###################################################
###########################################################################

class CapsuleAerodynamicGuidance(aerodynamics.AerodynamicGuidance):
    """
    Class to set the aerodynamic angles of the capsule (derived from the base class AerodynamicGuidance present in
    tudat).

    The class is only needed to initialize the aerodynamic guidance at the beginning of the propagation, where the
    sideslip angle and bank angle are set to zero, while the angle of attack is set as constant with a given value.
    During the propagation, the angles will be automatically updated and the aerodynamic coefficients computed from
    the database produced by the local inclination analysis based on the shape parameters provided.

    Attributes
    ----------
    bodies
    fixed_angle_of_attack

    Methods
    -------
    update_guidance(time)
        Function to compute the aerodynamic angles.
    """

    def __init__(self,
                 bodies: tudatpy.kernel.simulation.environment_setup.SystemOfBodies,
                 fixed_angle_of_attack: float):
        """
        Constructor for the CapsuleAerodynamicGuidance class.

        Parameters
        ----------
        bodies : tudatpy.kernel.simulation.environment_setup.SystemOfBodies
            System of bodies present in the simulation.
        fixed_angle_of_attack : float
            Angle of attack (constant) [rad].

        Returns
        -------
        none
        """
        # Call the base class constructor
        aerodynamics.AerodynamicGuidance.__init__(self)
        # Save arguments as attributes
        self.bodies = bodies
        self.fixed_angle_of_attack = fixed_angle_of_attack

    def updateGuidance(self,
                       currentTime: float):
        """
        Updates the attitude angles (default: all angles 0, angle-of-attack constant at given value).

        Parameters
        ----------
        currentTime : float
            Current epoch of the simulation (currently not used).

        Returns
        -------
        none
        """
        self.angle_of_attack = self.fixed_angle_of_attack
        self.bank_angle = 0.0
        self.sideslip_angle = 0.0

###########################################################################
# CREATE PROBLEM CLASS ####################################################
###########################################################################

class ShapeOptimizationProblem:
    """
    Class to initialize, simulate and optimize the Shape Optimization problem.

    The class is created specifically for this problem. This is done to provide better integration with Pagmo/Pygmo,
    where the optimization process (assignment 3) will be done. For the first two assignments, the presence of this
    class is not strictly needed, but it was chosen to create the code so that the same version could be used for all
    three assignments.

    Attributes
    ----------
    bodies
    integrator_settings
    propagator_settings
    capsule_density

    Methods
    -------
    get_last_run_propagated_state_history()
    get_last_run_dependent_variable_history()
    get_last_run_dynamics_simulator()
    fitness(shape_parameters)
    """

    def __init__(self,
                 bodies,
                 integrator_settings,
                 propagator_settings,
                 capsule_density,
                 decision_variable_range):
        """
        Constructor for the ShapeOptimizationProblem class.

        Parameters
        ----------
        bodies : tudatpy.kernel.simulation.environment_setup.SystemOfBodies
            System of bodies present in the simulation.
        integrator_settings : tudatpy.kernel.simulation.propagation_setup.integrator.IntegratorSettings
            Integrator settings to be provided to the dynamics simulator.
        propagator_settings : tudatpy.kernel.simulation.propagation_setup.propagator.MultiTypePropagatorSettings
            Propagator settings object.
        capsule_density : float
            Constant density of the vehicle.

        Returns
        -------
        none
        """
        # Set arguments as attributes
        self.bodies_function = lambda : bodies
        self.integrator_settings_function = lambda : integrator_settings
        self.propagator_settings_function = lambda : propagator_settings
        self.capsule_density = capsule_density
        self.decision_variable_range = decision_variable_range


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

    def get_bounds(self):

        return self.decision_variable_range

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

    def get_last_run_dynamics_simulator(self):
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
                shape_parameters):
        """
        Propagates the trajectory with the shape parameters given as argument.

        This function uses the shape parameters to set a new aerodynamic coefficient interface, subsequently propagating
        the trajectory. The fitness, currently set to zero, can be computed here: it will be used during the
        optimization process.

        Parameters
        ----------
        shape_parameters : list of floats
            List of shape parameters to be optimized.

        Returns
        -------
        fitness : float
            Fitness value (for optimization, see assignment 3).
        """
        bodies = self.bodies_function()
        integrator_settings = self.integrator_settings_function()
        propagator_settings = self.propagator_settings_function()

        # Delete existing capsule
        bodies.delete_body('Capsule')
        # Create new capsule with a new coefficient interface based on the current parameters, add it to the body system
        aerodynamic_analysis = add_capsule_to_body_system(bodies,
                                   shape_parameters,
                                   self.capsule_density)

        vehicle_mesh_surface = aerodynamics.get_local_inclination_total_vehicle_area(aerodynamic_analysis)
        [ vehicle_mesh_centroids, vehicle_mesh_surface_normals ] = aerodynamics.get_local_inclination_mesh(aerodynamic_analysis)
        print(vehicle_mesh_centroids)
        # Update propagation model with new body shape
        propagator_settings.recreate_state_derivative_models(bodies)

        # Create new aerodynamic guidance
        guidance_object = CapsuleAerodynamicGuidance(bodies,
                                                     shape_parameters[5])
        # Set aerodynamic guidance (this line links the CapsuleAerodynamicGuidance settings with the propagation)
        environment_setup.set_aerodynamic_guidance(guidance_object,
                                                   bodies.get_body('Capsule'))

        # Create simulation object and propagate dynamics
        dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
            bodies,
            integrator_settings,
            propagator_settings,
            print_dependent_variable_data = False)

        self.dynamics_simulator_function = lambda: dynamics_simulator

        # For the first two assignments, no computation of fitness is needed
        fitness = 0.0
        return [fitness]
