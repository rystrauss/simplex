"""Core components of the linear program solver.

This module provides an interface for constructing linear programs.

The user first creates a Solver, which is used to coordinate all aspects of the linear program.
After creating Solver, the user can add Variables, Constraints, and an Objective to the linear program.
Finally, the Solver can be used to solve the linear program with the Simplex method.

Author: Ryan Strauss
"""

from copy import copy
from typing import Final

import numpy as np

from simplex._simplex import solve_with_simplex, SolutionStatus


class Solver:
    """A utility for constructing and solving linear programs."""
    INFINITY: Final[float] = np.inf

    UNBOUNDED = SolutionStatus.UNBOUNDED
    INFEASIBLE = SolutionStatus.INFEASIBLE
    OPTIMAL = SolutionStatus.OPTIMAL

    def __init__(self):
        self._vars = dict()
        self._constraints = set()
        self._objective = None

    @property
    def variables(self):
        """An iterable containing the solver's variables."""
        return self._vars.keys()

    def add_variable(self, name, upper_bound=INFINITY):
        """Adds a variable to the linear program.

        Args:
            name: A unique identifier for variable.
            upper_bound: The variable's upper bound.

        Returns:
            The newly created variable.

        Raises:
            ValueError: if a variable with the provided name already exists for this solver
        """
        var = Variable(name, upper_bound)

        if var in self._vars:
            raise ValueError(f'this solver already has the variable: {name}')

        self._vars[var] = len(self._vars)
        return var

    def add_constraint(self, lower_bound, upper_bound):
        """Adds a constraint to the linear program.

        Args:
            lower_bound: The constraint's lower bound.
            upper_bound: The constraint's upper bound.

        Returns:
            The newly created constraint.
        """
        constraint = Constraint(lower_bound, upper_bound)
        self._constraints.add(constraint)
        return constraint

    def objective(self):
        """Creates the linear program's objective.

        Note that this method can be called more than once, but it will overwrite any previously
        created objectives.

        Returns:
            The newly created objective.
        """
        if self._objective is not None:
            print('\033[93mWarning: overwriting previously set objective.\033[0m')

        objective = Objective()
        self._objective = objective
        return objective

    def solve(self):
        """Solves the linear program.

        Returns:
            The program's solution status. Will be one of `Solver.OPTIMAL`, `Solver.UNBOUNDED`, and `Solver.INFEASIBLE`.
        """
        return solve_with_simplex(self._vars, self._constraints, self._objective)


class Variable:
    """A variable in a linear program.

    Attributes:
        name: The name of the variable.
        upper_bound: The upper bound for this variable's value.

    See Also:
        `Solver.add_variable`
    """

    def __init__(self, name, upper_bound=Solver.INFINITY):
        self.name = name
        self.upper_bound = upper_bound
        self._solution_value = None

    def standard_form(self):
        constraints = []

        if self.upper_bound != Solver.INFINITY:
            constraints.append(Constraint(0, self.upper_bound, {self: 1}))

        return constraints

    @property
    def solution_value(self):
        """The value of this variable in the optimal solution.

        Will be None if the linear program has not yet been solved, or the result of `Solver.solve` was not
        `Solver.OPTIMAL`.
        """
        return self._solution_value

    def __hash__(self):
        return hash(self.name)


class Constraint:
    """A constraint in a linear program.

    Attributes:
        lower_bound: The lower bound for this constraint's value.
        upper_bound: The upper bound for this constraint's value.
        coefficients: Optional. A dictionary mapping from variables to their coefficients for this constraint.

    See Also:
        `Solver.add_constraint`
    """

    def __init__(self, lower_bound, upper_bound, coefficients=None):
        if lower_bound > upper_bound:
            raise ValueError('lower bound cannot be greater than upper bound.')

        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        self._coefficients = dict() if coefficients is None else coefficients

    @property
    def lower_bound(self):
        """The lower bound for this variable's value."""
        return self._lower_bound

    @property
    def upper_bound(self):
        """The upper bound for this variable's value."""
        return self._upper_bound

    @property
    def coefficients(self):
        """A dictionary mapping from variables to their coefficients for this constraint."""
        return self._coefficients

    def set_coefficient(self, variable, value):
        """Sets the coefficient for a given variable in this constraint.

        Args:
            variable: The variable whose coefficient is being set.
            value: The value of the coefficient.

        Returns:
            None

        Raises:
            TypeError: if `variable` is not of the `Variable` type
        """
        if not isinstance(variable, Variable):
            raise TypeError('provided variable is not of the `Variable` type.')

        self._coefficients[variable] = value

    def standard_form(self):
        """Converts this constraint into the equivalent constraints that are in standard form.

        Returns:
            A list of constraints which are in standard form and, taken altogether, are equivalent to this constraint.
        """
        if self._lower_bound == -Solver.INFINITY and self._upper_bound != Solver.INFINITY:
            return [self]

        if self._lower_bound != -Solver.INFINITY and self._upper_bound == Solver.INFINITY:
            new_coefs = copy(self._coefficients)
            for key in new_coefs.keys():
                new_coefs[key] = -new_coefs[key]

            return [Constraint(-Solver.INFINITY, -self._lower_bound, new_coefs)]

        if self._lower_bound != -Solver.INFINITY and self._upper_bound != Solver.INFINITY:
            return [*Constraint(-Solver.INFINITY, self._upper_bound, copy(self._coefficients)).standard_form(),
                    *Constraint(self._lower_bound, Solver.INFINITY, copy(self._coefficients)).standard_form()]

        if self._lower_bound == -Solver.INFINITY and self._upper_bound == Solver.INFINITY:
            print('\033[93mWarning: using constraint with no bounds has no effect.\033[0m')
            return []


class Objective:
    """The objective of a linear program.

    See Also:
        `Solver.objective`
    """
    MAXIMIZATION: Final[str] = 'max'
    MINIMIZATION: Final[str] = 'min'

    def __init__(self):
        self._coefficients = dict()
        self._type = None
        self._solution_value = None

    @property
    def solution_value(self):
        """The value of the objective function in the optimal solution.

        Will be None if the linear program has not yet been solved, or the result of `Solver.solve` was not
        `Solver.OPTIMAL`.
        """
        return self._solution_value

    @property
    def type(self):
        """The type of this objective function, as either a minimization or maximization problem.

        Will be either `Objective.MINIMIZATION` or `Objective.MAXIMIZATION`, or None if the type has not
        yet been assigned.
        """
        return self._type

    @property
    def coefficients(self):
        """A dictionary mapping from variables to their coefficients for this objective."""
        return self._coefficients

    def set_coefficient(self, variable, value):
        """Sets the coefficient for a given variable in this objective.

        Args:
            variable: The variable whose coefficient is being set.
            value: The value of the coefficient.

        Returns:
            None

        Raises:
            TypeError: if `variable` is not of the `Variable` type
        """
        if not isinstance(variable, Variable):
            raise TypeError('provided variable is not of the `Variable` type.')

        self._coefficients[variable] = value

    def set_maximization(self):
        """Sets this objective to be maximized.

        Returns:
            None
        """
        self._type = self.MAXIMIZATION

    def set_minimization(self):
        """Sets this objective to be minimized.

        Returns:
            None
        """
        self._type = self.MINIMIZATION

    def standard_form(self):
        """Creates the standard form equivalent of this objective.

        Returns:
            An `Objective` that is equivalent to this one and is in standard form.
        """
        new = Objective()
        new._coefficients = copy(self._coefficients)
        new._type = self._type

        if new._type == self.MINIMIZATION:
            new._type = self.MAXIMIZATION
            for key in new._coefficients.keys():
                new._coefficients[key] = -new._coefficients[key]

        return new
