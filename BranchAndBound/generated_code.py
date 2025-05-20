import os
import time
import heapq
import numpy as np
from enum import Enum
from gurobipy import Env, Model, GRB, LinExpr
from utils.scp_data_reading import read_scp_raw_data, convert_scp_data_to_objects


class NodePriority(Enum):
    """Enum for different node evaluation priority strategies"""
    DEPTH_FIRST = 1  # Traditional depth-first (use negative depth as priority)
    BREADTH_FIRST = 2  # Traditional breadth-first (use depth as priority)
    BEST_BOUND = 3  # Best bound first (use node's bound as priority)
    HYBRID = 4  # Combination strategy (weighted score)
    DEPTH_ESTIMATE = 5  # Depth + estimate to optimal


class Node:
    """Node in the branch and bound tree"""
    node_counter = 0  # Static counter for node IDs

    def __init__(self, fixed_to_0, fixed_to_1, depth=0, bound=None):
        Node.node_counter += 1
        self.id = Node.node_counter
        self.fixed_to_0 = fixed_to_0.copy()  # Variables fixed to 0
        self.fixed_to_1 = fixed_to_1.copy()  # Variables fixed to 1
        self.depth = depth  # Depth in the B&B tree
        self.bound = bound  # Lower bound for this node
        self.branch_var = None  # Variable to branch on
        self.branch_val = None  # Value of the branching variable

    def __lt__(self, other):
        # Default comparison for priority queue
        # Lower bounds are better (minimization)
        if self.bound is None or other.bound is None:
            return self.depth > other.depth  # Default to depth-first if bounds not computed
        return self.bound < other.bound


class Result:
    def __init__(self, ov=None, variables=None):
        self._ov = ov if ov is not None else float('+inf')
        self._variables = variables if variables is not None else {}

    def get_ov(self):
        return self._ov

    def get_var(self):
        return self._variables


class FlexibleBranchAndBoundSCP:
    # Tolerance for pruning and numerical stability
    GAP_TOLERANCE = 1e-10

    def __init__(self, filename, priority_strategy=NodePriority.DEPTH_FIRST,
                 hybrid_weight=0.5, max_nodes=float('inf'), time_limit=float('inf')):
        """
        Initialize the branch and bound solver.

        Args:
            filename: SCP problem file
            priority_strategy: Strategy to prioritize nodes
            hybrid_weight: Weight for hybrid strategy (between 0 and 1)
            max_nodes: Maximum number of nodes to explore
            time_limit: Time limit in seconds
        """
        self.filename = filename
        self.priority_strategy = priority_strategy
        self.hybrid_weight = hybrid_weight
        self.max_nodes = max_nodes
        self.time_limit = time_limit

        basename = os.path.splitext(os.path.basename(filename))[0]
        log_file_path = os.getenv("LOG_FOLDER_PATH",
                                  "./logs") + f"/BranchAndBound_{basename}_{priority_strategy.name}.log"

        # Set up Gurobi environment
        self.env = Env(empty=True)
        self.env.setParam("LogFile", log_file_path)
        self.env.setParam("OutputFlag", 0)  # Suppress Gurobi output
        self.env.start()

        # Read problem data
        self.set_costs, self.user_coverage = read_scp_raw_data(filename)
        self.atm_list = convert_scp_data_to_objects(self.set_costs, self.user_coverage)

        # Initialize solution tracking
        self.best_solution = {atm.id: 0 for atm in self.atm_list}
        self.best_solution_value = float("+inf")
        self.upper_bound = float("+inf")
        self.lower_bound = 0.0

        # Statistics
        self.nodes_explored = 0
        self.nodes_pruned = 0
        self.nodes_created = 0
        self.time_to_solve = None
        self.solution_found_time = None
        self.first_solution_value = None
        self.first_solution_time = None

    def get_node_priority(self, node):
        """Calculate node priority based on selected strategy"""
        if node.bound is None:
            # If bound not calculated, use depth as default
            return -node.depth if self.priority_strategy == NodePriority.DEPTH_FIRST else node.depth

        if self.priority_strategy == NodePriority.DEPTH_FIRST:
            # Depth-first: prioritize deeper nodes (negative depth for min-heap)
            return (-node.depth, node.bound)

        elif self.priority_strategy == NodePriority.BREADTH_FIRST:
            # Breadth-first: prioritize shallower nodes
            return (node.depth, node.bound)

        elif self.priority_strategy == NodePriority.BEST_BOUND:
            # Best-bound: prioritize nodes with better bounds
            return (node.bound, node.depth)

        elif self.priority_strategy == NodePriority.HYBRID:
            # Hybrid: weighted combination of bound and depth
            # Normalize depth between 0 and 1 (assuming max depth = num variables)
            norm_depth = node.depth / len(self.atm_list)
            # Weight between bound and depth preference
            return (
                self.hybrid_weight * node.bound + (1 - self.hybrid_weight) * (-norm_depth),
                node.id  # Tie-breaker
            )

        elif self.priority_strategy == NodePriority.DEPTH_ESTIMATE:
            # Estimate of node's potential (bound + estimate to complete solution)
            remaining_vars = len(self.atm_list) - node.depth
            # Simple estimate: assume half of remaining vars will be 1
            estimate_to_complete = sum(sorted([atm.cost for atm in self.atm_list
                                               if atm.id not in node.fixed_to_0 and
                                               atm.id not in node.fixed_to_1])[:remaining_vars // 2])
            return (node.bound + estimate_to_complete, node.id)

        # Default
        return (node.bound, -node.depth)

    def solve(self, use_heuristic=True):
        """Solve the SCP problem using branch and bound"""
        if use_heuristic:
            self.greedy_heuristic()

        start_time = time.time()
        Node.node_counter = 0  # Reset node counter

        # Initialize with root node
        fixed_to_0 = {atm.id: False for atm in self.atm_list}
        fixed_to_1 = {atm.id: False for atm in self.atm_list}
        root = Node(fixed_to_0, fixed_to_1)

        # Solve LP relaxation for root node and set bound
        bound, frac_vars = self._solve_lp_relaxation(root.fixed_to_0, root.fixed_to_1)
        root.bound = bound

        # Initialize priority queue with root node
        priority_queue = []
        heapq.heappush(priority_queue, (self.get_node_priority(root), root))
        self.nodes_created += 1

        # Start branch and bound
        while priority_queue and self.nodes_explored < self.max_nodes:
            # Check time limit
            if time.time() - start_time > self.time_limit:
                print(f"Time limit of {self.time_limit} seconds reached.")
                break

            # Get next node based on priority
            _, current_node = heapq.heappop(priority_queue)
            self.nodes_explored += 1

            # Solve LP relaxation if bound not calculated
            if current_node.bound is None:
                current_node.bound, frac_vars = self._solve_lp_relaxation(
                    current_node.fixed_to_0, current_node.fixed_to_1
                )

            # Prune by bound
            if current_node.bound >= self.best_solution_value - self.GAP_TOLERANCE:
                self.nodes_pruned += 1
                continue

            # Check if solution is integer (no fractional variables)
            if not frac_vars:
                # Get solution
                sol, obj_val = self._get_solution(current_node.fixed_to_0, current_node.fixed_to_1)

                # Update best solution if better
                if obj_val < self.best_solution_value:
                    self.best_solution_value = obj_val
                    self.best_solution = sol.copy()
                    self.solution_found_time = time.time() - start_time

                    # Record first solution found
                    if self.first_solution_value is None:
                        self.first_solution_value = obj_val
                        self.first_solution_time = self.solution_found_time

                    print(f"New best solution found: {obj_val} at node {current_node.id}")
                continue

            # Branch on variable (select most fractional)
            branch_var, branch_val = min(frac_vars, key=lambda item: abs(0.5 - item[1]))

            # Create child nodes
            # Child 1: branch_var = 0
            child1 = Node(
                current_node.fixed_to_0.copy(),
                current_node.fixed_to_1.copy(),
                current_node.depth + 1
            )
            child1.fixed_to_0[branch_var] = True
            child1_bound, child1_frac = self._solve_lp_relaxation(child1.fixed_to_0, child1.fixed_to_1)

            # Only add if feasible and potentially better than current best
            if child1_bound is not None and child1_bound < self.best_solution_value - self.GAP_TOLERANCE:
                child1.bound = child1_bound
                child1.branch_var = branch_var
                child1.branch_val = 0
                heapq.heappush(priority_queue, (self.get_node_priority(child1), child1))
                self.nodes_created += 1
            else:
                self.nodes_pruned += 1

            # Child 2: branch_var = 1
            child2 = Node(
                current_node.fixed_to_0.copy(),
                current_node.fixed_to_1.copy(),
                current_node.depth + 1
            )
            child2.fixed_to_1[branch_var] = True
            child2_bound, child2_frac = self._solve_lp_relaxation(child2.fixed_to_0, child2.fixed_to_1)

            # Only add if feasible and potentially better than current best
            if child2_bound is not None and child2_bound < self.best_solution_value - self.GAP_TOLERANCE:
                child2.bound = child2_bound
                child2.branch_var = branch_var
                child2.branch_val = 1
                heapq.heappush(priority_queue, (self.get_node_priority(child2), child2))
                self.nodes_created += 1
            else:
                self.nodes_pruned += 1

        self.time_to_solve = time.time() - start_time
        print(f"\nSolve completed in {self.time_to_solve:.4f} seconds")
        print(f"Nodes explored: {self.nodes_explored}, created: {self.nodes_created}, pruned: {self.nodes_pruned}")
        print(f"Best solution value: {self.best_solution_value}")

        return Result(self.best_solution_value, self.best_solution)

    def _solve_lp_relaxation(self, fixed_to_0, fixed_to_1):
        """Solve LP relaxation for the given fixed variables"""
        m = Model(env=self.env)
        x = {}

        # Add variables
        for atm in self.atm_list:
            lb, ub = 0.0, 1.0
            if fixed_to_0[atm.id]:
                ub = 0.0
            if fixed_to_1[atm.id]:
                lb = 1.0
            x[atm.id] = m.addVar(lb=lb, ub=ub, obj=atm.cost, vtype=GRB.CONTINUOUS, name=f'x_{atm.id}')

        # Add constraints
        for user_id, cover in self.user_coverage.items():
            expr = LinExpr()
            for j in cover:
                expr.addTerms(1.0, x[j])
            m.addConstr(expr >= 1.0)

        m.modelSense = GRB.MINIMIZE
        m.update()
        m.optimize()

        # Check if feasible
        if m.status != GRB.OPTIMAL:
            m.dispose()
            return None, []

        # Get fractional variables
        frac_vars = [(index, value.x) for index, value in x.items()
                     if not np.isclose(value.x, 0.0, atol=self.GAP_TOLERANCE) and
                     not np.isclose(value.x, 1.0, atol=self.GAP_TOLERANCE)]

        obj_val = m.objVal
        m.dispose()

        return obj_val, frac_vars

    def _get_solution(self, fixed_to_0, fixed_to_1):
        """Get integer solution from fixed variables"""
        m = Model(env=self.env)
        x = {}

        # Add variables
        for atm in self.atm_list:
            lb, ub = 0, 1
            if fixed_to_0[atm.id]:
                lb = ub = 0
            elif fixed_to_1[atm.id]:
                lb = ub = 1
            x[atm.id] = m.addVar(lb=lb, ub=ub, obj=atm.cost, vtype=GRB.BINARY, name=f'x_{atm.id}')

        # Add constraints
        for user_id, cover in self.user_coverage.items():
            expr = LinExpr()
            for j in cover:
                expr.addTerms(1.0, x[j])
            m.addConstr(expr >= 1.0)

        m.modelSense = GRB.MINIMIZE
        m.update()
        m.optimize()

        # Extract solution
        sol = {i: int(round(v.x)) for i, v in x.items()}
        obj_val = m.objVal

        m.dispose()
        return sol, obj_val

    def greedy_heuristic(self):
        """Greedy heuristic to find initial feasible solution"""
        uncovered_users = set(self.user_coverage.keys())
        selected_atms = set()
        total_cost = 0

        while uncovered_users:
            best_atm = None
            best_score = float('-inf')

            for atm in self.atm_list:
                if atm.id in selected_atms:
                    continue  # Already selected

                # Users that would be newly covered
                covers = uncovered_users.intersection(atm.covered_users_ids)
                if not covers:
                    continue

                # Score: users covered per unit cost
                score = len(covers) / atm.cost

                if score > best_score:
                    best_score = score
                    best_atm = atm

            if best_atm is None:
                print("Warning: heuristic could not cover all users!")
                break

            selected_atms.add(best_atm.id)
            uncovered_users -= set(best_atm.covered_users_ids)
            total_cost += best_atm.cost

        # Store as initial best solution
        sol = {atm.id: 1 if atm.id in selected_atms else 0 for atm in self.atm_list}
        self.best_solution_value = total_cost
        self.best_solution = sol
        print(f"Heuristic solution found with cost: {total_cost}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # Example of comparing different node selection strategies
    filename = 'scp43.txt'

    strategies = [
        (NodePriority.DEPTH_FIRST, "Depth-First"),
        (NodePriority.BREADTH_FIRST, "Breadth-First"),
        (NodePriority.BEST_BOUND, "Best-Bound"),
        (NodePriority.HYBRID, "Hybrid (0.5)"),
        (NodePriority.DEPTH_ESTIMATE, "Depth + Estimate")
    ]

    results = []

    for strategy, name in strategies:
        print(f"\n===== Testing {name} Strategy =====")

        # With heuristic
        solver = FlexibleBranchAndBoundSCP(filename, priority_strategy=strategy)
        result = solver.solve(use_heuristic=True)

        results.append({
            'Strategy': name,
            'Heuristic': 'Yes',
            'Time': solver.time_to_solve,
            'Nodes Explored': solver.nodes_explored,
            'Nodes Created': solver.nodes_created,
            'First Solution Time': solver.first_solution_time,
            'First Solution Value': solver.first_solution_value,
            'Best Solution Value': result.get_ov()
        })

        print(f"{name} with heuristic: {solver.time_to_solve:.6f}s, Result: {result.get_ov()}")

        # Without heuristic
        solver = FlexibleBranchAndBoundSCP(filename, priority_strategy=strategy)
        result = solver.solve(use_heuristic=False)

        results.append({
            'Strategy': name,
            'Heuristic': 'No',
            'Time': solver.time_to_solve,
            'Nodes Explored': solver.nodes_explored,
            'Nodes Created': solver.nodes_created,
            'First Solution Time': solver.first_solution_time,
            'First Solution Value': solver.first_solution_value,
            'Best Solution Value': result.get_ov()
        })

        print(f"{name} without heuristic: {solver.time_to_solve:.6f}s, Result: {result.get_ov()}")

    # Print summary table
    print("\n===== Results Summary =====")
    print(f"{'Strategy':<20} {'Heuristic':<10} {'Time (s)':<10} {'Nodes':<10} {'Solution':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['Strategy']:<20} {'Yes' if r['Heuristic'] == 'Yes' else 'No':<10} "
              f"{r['Time']:<10.4f} {r['Nodes Explored']:<10} {r['Best Solution Value']:<10}")