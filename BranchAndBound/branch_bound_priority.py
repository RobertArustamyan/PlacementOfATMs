import os
import heapq
import time
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from gurobipy import Env, Model, GRB, LinExpr
from utils.scp_data_reading import read_scp_raw_data, convert_scp_data_to_objects


class SearchStrategy(Enum):
    DEPTH_FIRST = "depth_first"
    BREADTH_FIRST = "breadth_first"
    BEST_FIRST = "best_first"
    BEST_BOUND = "best_bound"
    HYBRID_DF_BF = "hybrid_df_bf"
    MOST_FRACTIONAL = "most_fractional"


@dataclass
class Node:
    """Represents a node in the branch-and-bound tree"""
    fixed_to_0: Dict[int, bool]
    fixed_to_1: Dict[int, bool]
    depth: int = 0
    lower_bound: float = 0.0
    fractional_vars: List[Tuple[int, float]] = field(default_factory=list)
    node_id: int = 0

    def __lt__(self, other):
        """For priority queue ordering - will be overridden by strategy"""
        return self.lower_bound < other.lower_bound


class Result:
    def __init__(self, oV=None, variables=None, time_taken=None, nodes_explored=None):
        self._oV = oV if oV is not None else float('+inf')
        self._variables = variables if variables is not None else {}
        self._time_taken = time_taken
        self._nodes_explored = nodes_explored

    def getOV(self):
        return self._oV

    def getVar(self):
        return self._variables

    def getTime(self):
        return self._time_taken

    def getNodesExplored(self):
        return self._nodes_explored


class BranchAndBoundSCPPriority:
    GAP_TOLERANCE = 1e-10

    def __init__(self, filename, strategy=SearchStrategy.DEPTH_FIRST, modelNo=0):
        self.modelNo = modelNo
        self.filename = filename
        self.strategy = strategy

        # Initialize Gurobi environment
        basename = os.path.splitext(os.path.basename(filename))[0]
        log_file_path = os.getenv("LOG_FOLDER_PATH", ".") + f"/BranchAndBound_{strategy.value}_{basename}.log"

        self.env = Env(empty=True)
        self.env.setParam("LogFile", log_file_path)
        self.env.setParam("OutputFlag", 0)
        self.env.start()

        # Load problem data
        self.set_costs, self.user_coverage = read_scp_raw_data(filename)
        self.atm_list = convert_scp_data_to_objects(self.set_costs, self.user_coverage)

        # Solution tracking
        self.best_solution = {atm.id: 0 for atm in self.atm_list}
        self.best_solution_value = float("+inf")
        self.nodes_explored = 0
        self.node_counter = 0

        # For hybrid strategy
        self.hybrid_switch_depth = 5
        self.initial_solutions_found = 0
        self.max_initial_solutions = 3

    def solve(self, use_heuristic=True, time_limit=300):
        """Main solve method with configurable strategy"""
        start_time = time.time()

        if use_heuristic:
            self.greedy_heuristic()

        # Initialize root node
        root = Node(
            fixed_to_0={atm.id: False for atm in self.atm_list},
            fixed_to_1={atm.id: False for atm in self.atm_list},
            depth=0,
            node_id=self.node_counter
        )
        self.node_counter += 1

        # Priority queue for nodes
        if self.strategy == SearchStrategy.DEPTH_FIRST:
            # Use stack (LIFO) for depth-first
            node_queue = [root]
        else:
            # Use priority queue for other strategies
            node_queue = [root]

        while node_queue and (time.time() - start_time) < time_limit:
            # Get next node based on strategy
            current_node = self._get_next_node(node_queue)
            if current_node is None:
                break

            self.nodes_explored += 1

            # Solve LP relaxation for current node
            result = self._solve_lp_relaxation(current_node)

            if result is None:  # Infeasible
                continue

            obj_val, solution, fractional_vars = result

            # Prune if bound is worse than best known solution
            if obj_val >= self.best_solution_value - self.GAP_TOLERANCE:
                continue

            # Check if solution is integer
            if not fractional_vars:
                # Found integer solution
                if obj_val < self.best_solution_value:
                    self.best_solution_value = obj_val
                    self.best_solution = {i: int(round(v)) for i, v in solution.items()}
                    self.initial_solutions_found += 1
                continue

            # Branch on fractional variable
            branch_var, branch_val = self._select_branching_variable(fractional_vars)

            # Create child nodes
            children = self._create_child_nodes(current_node, branch_var, obj_val, fractional_vars)

            # Add children to queue based on strategy
            self._add_nodes_to_queue(node_queue, children)

        solve_time = time.time() - start_time
        return Result(self.best_solution_value, self.best_solution, solve_time, self.nodes_explored)

    def _get_next_node(self, node_queue):
        """Get next node based on search strategy"""
        if not node_queue:
            return None

        if self.strategy == SearchStrategy.DEPTH_FIRST:
            return node_queue.pop()  # LIFO - stack behavior

        elif self.strategy == SearchStrategy.BREADTH_FIRST:
            return node_queue.pop(0)  # FIFO - queue behavior

        elif self.strategy in [SearchStrategy.BEST_FIRST, SearchStrategy.BEST_BOUND]:
            return heapq.heappop(node_queue)

        elif self.strategy == SearchStrategy.HYBRID_DF_BF:
            # Start with depth-first, then switch to breadth-first
            if (self.initial_solutions_found < self.max_initial_solutions and
                    any(node.depth <= self.hybrid_switch_depth for node in node_queue)):
                # Continue depth-first for nodes within switch depth
                deepest_nodes = [node for node in node_queue if node.depth <= self.hybrid_switch_depth]
                if deepest_nodes:
                    chosen_node = max(deepest_nodes, key=lambda x: x.depth)
                    node_queue.remove(chosen_node)
                    return chosen_node
            # Switch to best-first
            return heapq.heappop(node_queue) if node_queue else None

        elif self.strategy == SearchStrategy.MOST_FRACTIONAL:
            # Choose node with most fractional variables
            if node_queue:
                most_fractional = max(node_queue, key=lambda x: len(x.fractional_vars))
                node_queue.remove(most_fractional)
                return most_fractional

        return node_queue.pop() if node_queue else None

    def _add_nodes_to_queue(self, node_queue, children):
        """Add child nodes to queue based on strategy"""
        for child in children:
            if self.strategy == SearchStrategy.DEPTH_FIRST:
                node_queue.append(child)

            elif self.strategy == SearchStrategy.BREADTH_FIRST:
                node_queue.append(child)

            elif self.strategy in [SearchStrategy.BEST_FIRST, SearchStrategy.BEST_BOUND,
                                   SearchStrategy.HYBRID_DF_BF]:
                heapq.heappush(node_queue, child)

            elif self.strategy == SearchStrategy.MOST_FRACTIONAL:
                node_queue.append(child)

    def _solve_lp_relaxation(self, node):
        """Solve LP relaxation for a given node"""
        m = Model(env=self.env)
        x = {}

        # Create variables with bounds based on fixed variables
        for atm in self.atm_list:
            lb, ub = 0.0, 1.0
            if node.fixed_to_0[atm.id]:
                ub = 0.0
            if node.fixed_to_1[atm.id]:
                lb = 1.0
            x[atm.id] = m.addVar(lb=lb, ub=ub, obj=atm.cost, vtype=GRB.CONTINUOUS, name=f'x_{atm.id}')

        # Add coverage constraints
        for user_id, covering_sets in self.user_coverage.items():
            expr = LinExpr()
            for set_id in covering_sets:
                expr.addTerms(1.0, x[set_id])
            m.addConstr(expr >= 1.0)

        m.modelSense = GRB.MINIMIZE
        m.update()
        m.optimize()

        if m.status != GRB.OPTIMAL:
            m.dispose()
            return None

        obj_val = m.objVal
        solution = {i: v.x for i, v in x.items()}

        # Find fractional variables
        fractional_vars = [(i, v) for i, v in solution.items()
                           if not np.isclose(v, 0.0, atol=self.GAP_TOLERANCE) and
                           not np.isclose(v, 1.0, atol=self.GAP_TOLERANCE)]

        m.dispose()
        return obj_val, solution, fractional_vars

    def _select_branching_variable(self, fractional_vars):
        """Select which fractional variable to branch on"""
        if self.strategy == SearchStrategy.MOST_FRACTIONAL:
            # Branch on variable closest to 0.5
            return min(fractional_vars, key=lambda item: abs(0.5 - item[1]))
        else:
            # Default: branch on variable closest to 0.5
            return min(fractional_vars, key=lambda item: abs(0.5 - item[1]))

    def _create_child_nodes(self, parent, branch_var, parent_bound, fractional_vars):
        """Create child nodes by branching on the selected variable"""
        children = []

        # Child 1: fix variable to 1
        child1_fixed_0 = parent.fixed_to_0.copy()
        child1_fixed_1 = parent.fixed_to_1.copy()
        child1_fixed_1[branch_var] = True

        child1 = Node(
            fixed_to_0=child1_fixed_0,
            fixed_to_1=child1_fixed_1,
            depth=parent.depth + 1,
            lower_bound=parent_bound,
            fractional_vars=fractional_vars.copy(),
            node_id=self.node_counter
        )
        self.node_counter += 1
        children.append(child1)

        # Child 2: fix variable to 0
        child2_fixed_0 = parent.fixed_to_0.copy()
        child2_fixed_1 = parent.fixed_to_1.copy()
        child2_fixed_0[branch_var] = True

        child2 = Node(
            fixed_to_0=child2_fixed_0,
            fixed_to_1=child2_fixed_1,
            depth=parent.depth + 1,
            lower_bound=parent_bound,
            fractional_vars=fractional_vars.copy(),
            node_id=self.node_counter
        )
        self.node_counter += 1
        children.append(child2)

        return children

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
                    continue

                covers = uncovered_users.intersection(atm.covered_users_ids)
                if not covers:
                    continue

                score = len(covers) / atm.cost
                if score > best_score:
                    best_score = score
                    best_atm = atm

            if best_atm is None:
                break

            selected_atms.add(best_atm.id)
            uncovered_users -= set(best_atm.covered_users_ids)
            total_cost += best_atm.cost

        sol = {atm.id: 1 if atm.id in selected_atms else 0 for atm in self.atm_list}
        self.best_solution_value = total_cost
        self.best_solution = sol


def compare_strategies(filename, strategies=None, runs=5, time_limit=60):
    """Compare different search strategies"""
    if strategies is None:
        strategies = [SearchStrategy.DEPTH_FIRST, SearchStrategy.BREADTH_FIRST,
                      SearchStrategy.BEST_FIRST, SearchStrategy.HYBRID_DF_BF]

    results = {}

    for strategy in strategies:
        print(f"\nTesting strategy: {strategy.value}")
        times = []
        costs = []
        nodes = []

        for run in range(runs):
            solver = BranchAndBoundSCPPriority(filename, strategy)
            result = solver.solve(use_heuristic=True, time_limit=time_limit)

            times.append(result.getTime())
            costs.append(result.getOV())
            nodes.append(result.getNodesExplored())

            print(
                f"  Run {run + 1}: Time={result.getTime():.4f}s, Cost={result.getOV()}, Nodes={result.getNodesExplored()}")

        results[strategy.value] = {
            'avg_time': np.mean(times),
            'avg_cost': np.mean(costs),
            'avg_nodes': np.mean(nodes),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'best_cost': np.min(costs)
        }

    # Print comparison
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Strategy':<20} {'Avg Time':<10} {'Best Cost':<10} {'Avg Nodes':<12} {'Time Std':<10}")
    print("-" * 80)

    for strategy, stats in results.items():
        print(
            f"{strategy:<20} {stats['avg_time']:<10.4f} {stats['best_cost']:<10.0f} {stats['avg_nodes']:<12.0f} {stats['std_time']:<10.4f}")

    return results


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # Test single strategy
    print("Testing single strategy...")
    solver = BranchAndBoundSCPPriority('scp44.txt', SearchStrategy.DEPTH_FIRST)
    result = solver.solve(use_heuristic=True, time_limit=30)
    print(f"Result: Cost={result.getOV()}, Time={result.getTime():.4f}s, Nodes={result.getNodesExplored()}")

    # Compare strategies
    print("\nComparing strategies...")
    compare_results = compare_strategies('scp44.txt', runs=3, time_limit=30)