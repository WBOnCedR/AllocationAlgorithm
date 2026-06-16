import numpy as np
import cvxpy as cp
import time
from datetime import datetime


class AllocationOptimizer:
    """
    Constrained optimization algorithm for the allocation of operational
    modules to resources under scarcity, as described in
    "Development of a Constrained Optimization Algorithm for the Allocation
    of Operational Modules in Resource Scarcity Scenarios".

    Two stages:
      1. Module score computation: a weight w_i is assigned to each module,
         inversely proportional to its excess supply relative to demand.
      2. Allocation: a Mixed-Integer Quadratic Program assigns resources to
         modules, minimizing the squared deviation from a proportional
         target T_j(alpha) for each resource, subject to demand
         satisfaction, availability, and participation constraints.
    """

    def __init__(self, availability_matrix, demand_vector, alpha=0.7):
        self.availability_matrix = availability_matrix
        self.demand_vector = demand_vector
        self.alpha = alpha

        self._module_weights = None
        self._targets = None
        self._allocation_matrix = None
        self._weighted_allocation = None
        self._objective_value = None
        self._status = None

    @property
    def availability_matrix(self):
        """Availability matrix S, shape (n_modules, n_resources)."""
        return self._availability_matrix

    @availability_matrix.setter
    def availability_matrix(self, value):
        self._availability_matrix = np.asarray(value)
        self._invalidate_results()

    @property
    def demand_vector(self):
        """Demand vector D, shape (n_modules, 1)."""
        return self._demand_vector

    @demand_vector.setter
    def demand_vector(self, value):
        self._demand_vector = np.asarray(value).reshape(-1, 1)
        self._invalidate_results()

    @property
    def alpha(self):
        """Trade-off parameter between egalitarian (0) and proportional (1) targets."""
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value
        self._targets = None
        self._allocation_matrix = None
        self._weighted_allocation = None

    def _invalidate_results(self):
        self._module_weights = None
        self._targets = None
        self._allocation_matrix = None
        self._weighted_allocation = None
        self._objective_value = None
        self._status = None

    @property
    def n_modules(self):
        return self.availability_matrix.shape[0]

    @property
    def n_resources(self):
        return self.availability_matrix.shape[1]

    @property
    def module_availability(self):
        """Total availability per module: sum over resources (s_i)."""
        return np.sum(self.availability_matrix, axis=1).reshape(-1, 1)

    @property
    def resource_availability(self):
        """Total availability per resource: sum over modules (S_j)."""
        return np.sum(self.availability_matrix, axis=0)

    @property
    def active_resources(self):
        """Number of resources with at least one availability (k_active)."""
        return int(np.sum(self.resource_availability > 0))

    @property
    def total_demand(self):
        """Total demand of the system (D_tot)."""
        return float(np.sum(self.demand_vector))

    @property
    def total_availability(self):
        """Total availability of the system (S_tot)."""
        return float(np.sum(self.availability_matrix))

    @property
    def module_weights(self):
        """Weight w_i assigned to each module (computed on first access)."""
        if self._module_weights is None:
            self.compute_module_weights()
        return self._module_weights

    def compute_module_weights(self):
        """
        Computes the module score vector W as a logarithmic function of the
        excess supply relative to demand:

            w_i = 0                                   if s_i - d_i <= 0
            w_i = log(s_i - d_i) / max(log(s_i - d_i)) otherwise

        Returns the weight vector, shape (n_modules, 1).
        """
        excess = self.module_availability - self.demand_vector
        max_log_excess = np.max(np.log(excess, where=excess > 0, out=np.zeros_like(excess, dtype=float)))
        self._module_weights = np.where(
            excess <= 0,
            0,
            np.round(np.log(excess) / max_log_excess, 4),
        )
        return self._module_weights

    @property
    def targets(self):
        """Proportional target T_j(alpha) for each resource."""
        return self._targets

    @property
    def allocation_matrix(self):
        """Optimal allocation matrix X, shape (n_modules, n_resources)."""
        return self._allocation_matrix

    @property
    def weighted_allocation(self):
        """Weighted load assigned to each resource: sum_i w_i * x_ij."""
        return self._weighted_allocation

    @property
    def objective_value(self):
        """Final value of the loss function."""
        return self._objective_value

    @property
    def status(self):
        """Solver status of the last allocation run."""
        return self._status

    def solve(self, max_time=3600, verbose=True):
        """
        Solves the Mixed-Integer Quadratic Program that allocates resources
        to modules, minimizing the squared deviation from the proportional
        target T_j(alpha), subject to:
          - Demand satisfaction: each module receives exactly d_i resources;
          - Availability: a resource can only be assigned where available;
          - Participation guarantee: every resource with at least one
            availability receives at least one assignment.

        Returns the allocation matrix, or None if no optimal solution is found.
        """
        start_global = time.time()
        S = self.availability_matrix
        D = self.demand_vector
        W = self.module_weights
        n, k = S.shape

        if verbose:
            print("=" * 80)
            print(f"STARTING OPTIMIZATION - {datetime.now().strftime('%H:%M:%S')}")
            print(f"Social trade-off parameter (alpha): {self.alpha}")
            print("=" * 80)

        available_idx = np.argwhere(S == 1)
        x_vars = {
            tuple(idx): cp.Variable(boolean=True, name=f"x_{idx[0]}_{idx[1]}")
            for idx in available_idx
        }

        constraints = []

        for i in range(n):
            vars_i = [x_vars[(i, j)] for j in range(k) if (i, j) in x_vars]
            if vars_i:
                constraints.append(cp.sum(vars_i) == D[i, 0])
            elif D[i, 0] > 0:
                if verbose:
                    print(f"ERROR: cannot satisfy demand for module {i}. No availability.")
                return None

        for j in range(k):
            vars_j = [x_vars[(i, j)] for i in range(n) if (i, j) in x_vars]
            if len(vars_j) > 0:
                constraints.append(cp.sum(vars_j) >= 1)

        resource_availability = self.resource_availability
        total_demand = self.total_demand
        total_availability = self.total_availability
        active_resources = self.active_resources

        obj_terms = []
        targets = np.zeros(k)

        for j in range(k):
            if resource_availability[j] > 0:
                target_j = (1 - self.alpha) * (total_demand / active_resources) + self.alpha * (
                    total_demand * (resource_availability[j] / total_availability)
                )
                targets[j] = target_j

                weighted_load_j = cp.sum(
                    [W[i, 0] * x_vars[(i, j)] for i in range(n) if (i, j) in x_vars]
                )

                obj_terms.append(cp.square(target_j - weighted_load_j))

        problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)

        scip_params = {
            "limits/time": max_time,
            "display/verblevel": 4 if verbose else 0,
            "display/freq": 1,
            "limits/gap": 0.001,
        }

        try:
            problem.solve(solver=cp.SCIP, scip_params=scip_params, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"SOLVER ERROR: {e}")
            return None

        self._status = problem.status
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            allocation = np.zeros((n, k), dtype=int)
            for (i, j), var in x_vars.items():
                if var.value is not None:
                    allocation[i, j] = int(np.round(var.value))

            self._allocation_matrix = allocation
            self._targets = targets
            self._objective_value = problem.value
            self._weighted_allocation = np.array(
                [np.sum(allocation[:, j] * W[:, 0]) for j in range(k)]
            )

            if verbose:
                print(f"\n[3/4] PERFORMANCE SUMMARY")
                print(f"   - Status: {problem.status}")
                print(f"   - Final loss: {problem.value:.6f}")
                print(f"   - Total demand: {total_demand} | Total availability: {total_availability}")

                print(f"\n[4/4] LOAD DISTRIBUTION")
                print(f"{'Res.':<8} | {'Avail.':<8} | {'Target':<8} | {'Assigned':<10} | {'Gap'}")
                print("-" * 55)
                for j in range(k):
                    if resource_availability[j] > 0:
                        print(
                            f"{j:<8} | {resource_availability[j]:<8.1f} | {targets[j]:<8.2f} | "
                            f"{self._weighted_allocation[j]:<10.2f} | {self._weighted_allocation[j] - targets[j]:>6.2f}"
                        )

                print(f"\nCOMPLETED in {time.time() - start_global:.2f}s")
                print("=" * 80)

            return allocation

        else:
            if verbose:
                print(f"NO OPTIMAL SOLUTION FOUND. Status: {problem.status}")
            return None
