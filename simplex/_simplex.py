"""This module contains an implementation of the basic Simplex algorithm.

This module is only meant to be used by the `Solver` class and should not be used directly by the user.

Author: Ryan Strauss
"""

from enum import Enum, auto

import numpy as np


class SolutionStatus(Enum):
    """An enum with the three possible outcomes for a linear program."""
    OPTIMAL = auto()
    UNBOUNDED = auto()
    INFEASIBLE = auto()


def get_initial_tableau(variables, constraints, objective):
    """Produces the initial tableau for a given linear program.

    Args:
        variables: The variables of the LP.
        constraints: The constraints of the LP.
        objective: The objective of the LP.

    Returns:
        THe linear program's initial tableau.
    """
    num_vars = len(variables)
    num_constraints = len(constraints)

    b_column = np.zeros((num_constraints + 1, 1), dtype=np.float)
    coef_matrix = np.zeros((num_constraints + 1, num_vars), dtype=np.float)

    for i, constraint in enumerate(constraints):
        b_column[i, 0] = constraint.upper_bound
        for var, coef in constraint.coefficients.items():
            coef_matrix[i, variables[var]] = coef

    for var, coef in objective.coefficients.items():
        coef_matrix[-1, variables[var]] = -coef

    tableau = np.hstack((
        coef_matrix,
        np.eye(num_constraints + 1),
        b_column
    ))

    return tableau


def perform_pivot(tableau, row, col):
    """Performs a pivot operation on a tableau.

    Args:
        tableau: The tableau to be operated on.
        row: The row of the pivot operation.
        col: The column of the pivot operation.

    Returns:
        None
    """
    n = tableau.shape[0]
    s = np.sign(tableau[row, col])

    tableau[row, :] *= s

    for i in range(n):
        if i != row:
            tableau[i, :] = \
                s * (tableau[row, col] * tableau[i, :] - tableau[i, col] * tableau[row, :]) / tableau[-1, -2]


def standardize(constraints, objective):
    """Converts the constraints and objective to standard form.

    Args:
        constraints: The constraints to be converted.
        objective: The objective to be converted.

    Returns:
        The new constraints and objective.
    """
    standard_constraints = []
    for c in constraints:
        standard_constraints.extend(c.standard_form())

    standard_objective = objective.standard_form()

    return standard_constraints, standard_objective


def phase_one_pivot_position(tableau):
    """Calculates the pivot position for phase one of the Simplex algorithm.

    Args:
        tableau: The tableau to be analyzed.

    Returns:
        (pivot_row, pivot_col)
    """
    prow = np.argmin(tableau[:-1, -1].flatten())
    pcol = 0

    while tableau[prow, pcol] >= 0:
        pcol += 1

    if pcol >= tableau.shape[1] - 1:
        return None, None

    return prow, pcol


def phase_two_pivot_position(tableau):
    """Calculates the pivot position for phase two of the Simplex algorithm.

    Args:
        tableau: The tableau to be analyzed.

    Returns:
        (pivot_row, pivot_col)
    """
    prow = 0
    pcol = 0

    while tableau[-1, pcol] >= 0:
        pcol += 1

    assert pcol < tableau.shape[1] - 1

    min_pos_bratio = np.inf

    for i, row in enumerate(tableau[:-1]):
        bratio = 0 if row[pcol] == 0 else row[-1] / row[pcol]
        if bratio > 0 and bratio < min_pos_bratio:
            min_pos_bratio = bratio
            prow = i

    if min_pos_bratio == np.inf:
        return None, None

    return prow, pcol


def simplex(tableau):
    """Executes the simple version of the Simplex algorithm.

    Args:
        tableau: The initial tableau to be solved.

    Returns:
        One of the outcomes contained in `SolutionStatus`.
    """
    # Phase I
    while tableau[:-1, -1].min() < 0:
        prow, pcol = phase_one_pivot_position(tableau)

        if prow is None:
            return SolutionStatus.INFEASIBLE

        perform_pivot(tableau, prow, pcol)

    # Phase II
    while tableau[-1, :-1].min() < 0:
        prow, pcol = phase_two_pivot_position(tableau)

        if prow is None:
            return SolutionStatus.UNBOUNDED

        perform_pivot(tableau, prow, pcol)

    return SolutionStatus.OPTIMAL


def get_results(tableau, variables, objective):
    """Extracts solution values from the final tableau.

    The solution values of the provided variables and objective are directly updated -- nothing is returned.

    Args:
        tableau: The final tableau for an optimal solution.
        variables: The variables to be updated.
        objective: The objective to be updated.

    Returns:
        None.
    """
    objective._solution_value = tableau[-1, -1] / tableau[-1, -2]

    for var in variables:
        i = variables[var]
        if np.count_nonzero(tableau[:, i]) > 1:
            continue

        j = np.argmax(tableau[:, i])
        var._solution_value = tableau[j, -1] / tableau[j, i]


def solve_with_simplex(variables, constraints, objective):
    """Solves a linear program using the Simplex algorithm.

    The arguments to this method should come directly from a `Solver`.

    Args:
        variables: The variables of the LP.
        constraints: The constraints of the LP.
        objective: The objective of the LP.

    Returns:
        One of the outcomes contained in `SolutionStatus`.
    """
    constraints, standard_objective = standardize(constraints, objective)
    tableau = get_initial_tableau(variables, constraints, standard_objective)
    outcome = simplex(tableau)

    if outcome == SolutionStatus.OPTIMAL:
        get_results(tableau, variables, objective)

    return outcome
