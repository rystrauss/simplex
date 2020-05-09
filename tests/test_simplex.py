import pytest

from simplex import Solver

TIMEOUT = 5


@pytest.mark.timeout(TIMEOUT)
def test_lp1():
    solver = Solver()

    x = solver.add_variable('x', Solver.INFINITY)
    y = solver.add_variable('y', Solver.INFINITY)

    constraint0 = solver.add_constraint(-Solver.INFINITY, 14)
    constraint0.set_coefficient(x, 1)
    constraint0.set_coefficient(y, 2)

    constraint1 = solver.add_constraint(0, Solver.INFINITY)
    constraint1.set_coefficient(x, 3)
    constraint1.set_coefficient(y, -1)

    constraint2 = solver.add_constraint(-Solver.INFINITY, 2)
    constraint2.set_coefficient(x, 1)
    constraint2.set_coefficient(y, -1)

    objective = solver.objective()
    objective.set_coefficient(x, 3)
    objective.set_coefficient(y, 4)

    status = solver.solve()

    assert status == Solver.OPTIMAL
    assert objective.solution_value == 34.0
    assert x.solution_value == 6.0
    assert y.solution_value == 4.0


@pytest.mark.timeout(TIMEOUT)
def test_lp2():
    solver = Solver()

    x = solver.add_variable('x', Solver.INFINITY)
    y = solver.add_variable('y', Solver.INFINITY)

    constraint0 = solver.add_constraint(-Solver.INFINITY, 480)
    constraint0.set_coefficient(x, 5)
    constraint0.set_coefficient(y, 15)

    constraint1 = solver.add_constraint(-Solver.INFINITY, 160)
    constraint1.set_coefficient(x, 4)
    constraint1.set_coefficient(y, 4)

    constraint2 = solver.add_constraint(-Solver.INFINITY, 1190)
    constraint2.set_coefficient(x, 35)
    constraint2.set_coefficient(y, 20)

    objective = solver.objective()
    objective.set_coefficient(x, 13)
    objective.set_coefficient(y, 23)

    status = solver.solve()

    assert status == Solver.OPTIMAL
    assert objective.solution_value == 800
    assert x.solution_value == 12
    assert y.solution_value == 28


@pytest.mark.timeout(TIMEOUT)
def test_lp3():
    solver = Solver()

    x = solver.add_variable('x', 50)
    y = solver.add_variable('y', 40)

    constraint0 = solver.add_constraint(60, Solver.INFINITY)
    constraint0.set_coefficient(x, 1)
    constraint0.set_coefficient(y, 1)

    constraint1 = solver.add_constraint(-Solver.INFINITY, 80)
    constraint1.set_coefficient(x, 1)
    constraint1.set_coefficient(y, 1)

    objective = solver.objective()
    objective.set_coefficient(x, 300)
    objective.set_coefficient(y, 150)

    status = solver.solve()

    assert status == Solver.OPTIMAL
    assert objective.solution_value == 19500
    assert x.solution_value == 50
    assert y.solution_value == 30


@pytest.mark.timeout(TIMEOUT)
def test_lp4():
    solver = Solver()

    x = solver.add_variable('x', Solver.INFINITY)
    y = solver.add_variable('y', Solver.INFINITY)

    constraint0 = solver.add_constraint(11, Solver.INFINITY)
    constraint0.set_coefficient(x, 1)
    constraint0.set_coefficient(y, 1)

    constraint1 = solver.add_constraint(-Solver.INFINITY, 5)
    constraint1.set_coefficient(x, 1)
    constraint1.set_coefficient(y, -1)

    constraint2 = solver.add_constraint(0, 0)
    constraint2.set_coefficient(x, -1)
    constraint2.set_coefficient(y, -1)

    constraint3 = solver.add_constraint(35, Solver.INFINITY)
    constraint3.set_coefficient(x, 7)
    constraint3.set_coefficient(y, -12)

    objective = solver.objective()
    objective.set_coefficient(x, 10)
    objective.set_coefficient(y, 11)

    status = solver.solve()

    assert status == Solver.UNBOUNDED
    assert objective.solution_value is None
    assert x.solution_value is None
    assert y.solution_value is None
