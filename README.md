# Allocation Algorithm

Constrained optimization algorithm for allocating operational modules to a
pool of resources under scarcity, based on individually provided
availabilities. See [the paper](about:blank) for the full mathematical
formulation.

## Overview

Traditional Greedy or Round-Robin allocation treats every module as
identical, ignoring how scarce its supply is. This algorithm instead:

1. **Scores each module** with a weight $w_i$, a logarithmic function of its
   excess supply over demand — modules with little slack score low, modules
   with abundant supply score close to 1.
2. **Allocates resources to modules** by solving a Mixed-Integer Quadratic
   Program that minimizes the squared deviation from a *proportional
   target* $T_j(\alpha)$ per resource — the fair share of the total demand
   each resource should receive, blending an egalitarian and a
   merit-based (availability-proportional) target via $\alpha \in [0,1]$ —
   subject to demand satisfaction, availability, and participation
   constraints.

## `AllocationClass.py`

Exposes a single class, `AllocationOptimizer`, that wraps both stages above.

**Properties** (inputs):
- `availability_matrix` — binary matrix, modules × resources.
- `demand_vector` — required resources per module.
- `alpha` — trade-off between egalitarian and proportional targets.

**Properties** (intermediate / derived results, available after the
corresponding computation has run):
- `module_weights` — the score $w_i$ per module.
- `targets` — the proportional target $T_j(\alpha)$ per resource.
- `allocation_matrix` — the optimal binary allocation.
- `weighted_allocation` — the weighted load assigned to each resource.
- `objective_value`, `status` — solver outcome.
- Derived inputs such as `module_availability`, `resource_availability`,
  `active_resources`, `total_demand`, `total_availability`.

**Methods:**
- `compute_module_weights()` — runs the module scoring stage.
- `solve(max_time=3600, verbose=True)` — runs the MIQP allocation stage
  (via CVXPY + SCIP) and populates the result properties above.

```python
from AllocationClass import AllocationOptimizer

optimizer = AllocationOptimizer(availability_matrix, demand_vector, alpha=0.7)
weights = optimizer.module_weights      # module score w_i
allocation = optimizer.solve()          # MIQP allocation matrix

optimizer.targets                       # proportional target T_j per resource
optimizer.weighted_allocation           # weighted load assigned per resource
optimizer.objective_value               # final loss value
optimizer.status                        # solver status
```

## Setup

```bash
pip install -r requirements.txt
```

SCIP (via `pyscipopt`) is used as the MIQP solver through CVXPY.
