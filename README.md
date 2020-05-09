# Simplex Algorithm

The `simplex` package contained in this repository provides an interface for constructing linear programs
and solving them with the [Simplex algorithm](https://en.wikipedia.org/wiki/Simplex_algorithm).

## Usage

Consider the following LP:

**Maximize 3x + 4y subject to the following constraints:**

x + 2y ≤ 14

3x - y ≥ 0

x - y ≤ 2


The code below demonstrates how to construct and solve this LP using the `simplex` package.

For simplicity, this solver assumes that the objective is to be maximized and does not provide the option to set
explicit lower bounds on variables.

```python
from simplex import Solver

# Create the solver
solver = Solver()

# Create variables
x = solver.add_variable('x', 0, Solver.INFINITY)
y = solver.add_variable('y', 0, Solver.INFINITY)

# Constraint 0: x + 2y <= 14.
constraint0 = solver.add_constraint(-Solver.INFINITY, 14)
constraint0.set_coefficient(x, 1)
constraint0.set_coefficient(y, 2)

# Constraint 1: 3x - y >= 0.
constraint1 = solver.add_constraint(0, Solver.INFINITY)
constraint1.set_coefficient(x, 3)
constraint1.set_coefficient(y, -1)

# Constraint 2: x - y <= 2.
constraint2 = solver.add_constraint(-Solver.INFINITY, 2)
constraint2.set_coefficient(x, 1)
constraint2.set_coefficient(y, -1)

# Objective function: 3x + 4y.
objective = solver.objective()
objective.set_coefficient(x, 3)
objective.set_coefficient(y, 4)
objective.set_maximization()

# Solve the linear program
status = solver.solve()

print('Solution status:', status)
print('Objective value:', objective.solution_value)
print('x value:', x.solution_value)
print('y value:', y.solution_value)
```
