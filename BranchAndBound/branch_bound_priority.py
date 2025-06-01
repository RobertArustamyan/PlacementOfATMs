import os
import heapq
import time
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from gurobipy import Env, Model, GRB, LinExpr
from utils.scp_data_reading import read_scp_raw_data, convert_scp_data_to_objects
from algorithms.heuristics_file import AdaptiveManager, HeuristicSolution, ConstructiveHeuristics, \
    LocalSearchHeuristics, BoundingHeuristics


class SearchStrategy(Enum):
    DEPTH_FIRST = "depth_first"
    BREADTH_FIRST = "breadth_first"
    BEST_FIRST = "best_first"
    BEST_BOUND = "best_bound"
    HYBRID_DF_BF = "hybrid_df_bf"
    MOST_FRACTIONAL = "most_fractional"


class HeuristicStrategy(Enum):
    NONE = "none"
    BASIC = "basic"  # Just constructive heuristics
    INTERMEDIATE = "intermediate"  # Constructive + local search
    ADVANCED = "advanced"  # Full adaptive manager
    CUSTOM = "custom"  # Custom heuristic selection


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
    def __init__(self, oV=None, variables=None, time_taken=None, nodes_explored=None,
                 heuristic_info=None):
        self._oV = oV if oV is not None else float('+inf')
        self._variables = variables if variables is not None else {}
        self._time_taken = time_taken
        self._nodes_explored = nodes_explored
        self._heuristic_info = heuristic_info or {}

    def getOV(self):
        return self._oV

    def getVar(self):
        return self._variables

    def getTime(self):
        return self._time_taken

    def getNodesExplored(self):
        return self._nodes_explored

    def getHeuristicInfo(self):
        return self._heuristic_info

    def getGap(self):
        return self._heuristic_info.get('gap', float('inf'))

    def isOptimal(self):
        return self._heuristic_info.get('is_optimal', False)


class BranchAndBoundSCPPriority:
    GAP_TOLERANCE = 1e-10

    def __init__(self, filename, strategy=SearchStrategy.DEPTH_FIRST,
                 heuristic_strategy=HeuristicStrategy.NONE, modelNo=0):
        self.modelNo = modelNo
        self.filename = filename
        self.strategy = strategy
        self.heuristic_strategy = heuristic_strategy

        self.time_allocations = {
            'initial': 0.10,  # 10% for initial heuristics
            'main': 0.80,  # 80% for main B&B
            'final': 0.10  # 10% for final intensification
        }

        # Initialize Gurobi environment
        basename = os.path.splitext(os.path.basename(filename))[0]
        log_file_path = os.getenv("LOG_FOLDER_PATH",
                                  ".") + f"/BranchAndBound_{strategy.value}_{heuristic_strategy.value}_{basename}.log"

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

        # Heuristic components
        self.adaptive_manager = None
        self.constructive_heuristics = None
        self.local_search_heuristics = None
        self.bounding_heuristics = None

        # Heuristic tracking
        self.nodes_since_last_heuristic = 0
        self.last_heuristic_time = 0
        self.heuristic_solutions = []
        self.heuristic_calls = 0
        self.heuristic_improvements = 0
        self.total_heuristic_time = 0

        # Performance tracking
        self.heuristic_performance = {
            'initial_phase': {},
            'periodic_improvements': {},
            'final_phase': {},
            'method_calls': {},
            'best_methods': []
        }

    def _initialize_heuristics(self, time_budget=300):
        """Initialize heuristic components based on strategy"""
        if self.heuristic_strategy == HeuristicStrategy.NONE:
            return

        print(f"Initializing heuristics with strategy: {self.heuristic_strategy.value}")
        print(f"Time allocations: Initial={self.time_allocations['initial']:.1%}, "
              f"Main={self.time_allocations['main']:.1%}, "
              f"Final={self.time_allocations['final']:.1%}")

        if self.heuristic_strategy == HeuristicStrategy.BASIC:
            # Only constructive heuristics
            self.constructive_heuristics = ConstructiveHeuristics(self.atm_list, self.user_coverage)

        elif self.heuristic_strategy == HeuristicStrategy.INTERMEDIATE:
            # Constructive + local search
            self.constructive_heuristics = ConstructiveHeuristics(self.atm_list, self.user_coverage)
            self.local_search_heuristics = LocalSearchHeuristics(self.atm_list, self.user_coverage)
            self.bounding_heuristics = BoundingHeuristics(self.atm_list, self.user_coverage)

        elif self.heuristic_strategy == HeuristicStrategy.ADVANCED:
            # Full adaptive manager
            self.adaptive_manager = AdaptiveManager(
                self.atm_list,
                self.user_coverage,
                time_budget=time_budget
            )
            # Update time allocations in the adaptive manager
            self.adaptive_manager.phase_allocations = self.time_allocations

        elif self.heuristic_strategy == HeuristicStrategy.CUSTOM:
            # Custom selection - initialize all for flexibility
            self.constructive_heuristics = ConstructiveHeuristics(self.atm_list, self.user_coverage)
            self.local_search_heuristics = LocalSearchHeuristics(self.atm_list, self.user_coverage)
            self.bounding_heuristics = BoundingHeuristics(self.atm_list, self.user_coverage)

    def solve(self, use_heuristic=True, time_limit=300):
        """Main solve method with configurable strategy"""
        start_time = time.time()

        self.current_time_limit = time_limit

        # Initialize best lower bound for gap calculation
        best_lower_bound = 0.0  # Since all costs are non-negative in set cover

        # Initialize heuristics if requested
        if use_heuristic and self.heuristic_strategy != HeuristicStrategy.NONE:
            self._initialize_heuristics(time_limit)

            # Phase 1: Initial heuristic burst
            initial_phase_time = time_limit * self.time_allocations['initial']
            print(f"Starting initial heuristics phase with {initial_phase_time:.1f}s budget")
            initial_solutions = self._run_initial_heuristics(initial_phase_time)

            if initial_solutions:
                best_initial = min(initial_solutions, key=lambda x: x.cost)
                if best_initial.cost < self.best_solution_value:
                    self.best_solution_value = best_initial.cost
                    self.best_solution = best_initial.solution
                    print(f"Initial heuristic found solution with cost: {best_initial.cost}")

        # Initialize root node
        root = Node(
            fixed_to_0={atm.id: False for atm in self.atm_list},
            fixed_to_1={atm.id: False for atm in self.atm_list},
            depth=0,
            node_id=self.node_counter
        )
        self.node_counter += 1

        # Initialize node queue based on strategy
        if self.strategy in [SearchStrategy.BEST_FIRST, SearchStrategy.BEST_BOUND, SearchStrategy.HYBRID_DF_BF]:
            # Use heap for priority-based strategies
            node_queue = [root]
            heapq.heapify(node_queue)
        else:
            # Use regular list for stack/queue-based strategies
            node_queue = [root]

        # Main B&B loop
        bb_start_time = time.time()
        print(f"Starting main B&B phase with {time_limit * self.time_allocations['main']:.1f}s budget")

        while node_queue and (time.time() - start_time) < time_limit:
            # Get next node based on strategy
            current_node = self._get_next_node(node_queue)
            if current_node is None:
                break

            self.nodes_explored += 1
            self.nodes_since_last_heuristic += 1

            # Periodic heuristic improvements
            if (use_heuristic and self.heuristic_strategy != HeuristicStrategy.NONE and
                    self._should_run_periodic_heuristic(time.time() - start_time)):
                self._run_periodic_heuristic_improvement(time.time() - start_time)

            # Solve LP relaxation for current node
            result = self._solve_lp_relaxation(current_node)

            if result is None:  # Infeasible
                continue

            obj_val, solution, fractional_vars = result

            # Update best lower bound (for gap calculation)
            # The best lower bound is the minimum LP relaxation value among all active nodes
            if self.nodes_explored == 1:  # Root node
                best_lower_bound = obj_val
            elif node_queue:  # If there are nodes in the queue
                # For exact calculation, we'd need to track bounds for all active nodes
                # Here we use a simple approximation: current node's bound
                if obj_val < self.best_solution_value:
                    best_lower_bound = max(best_lower_bound, obj_val)

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
                    print(f"New best integer solution found: {obj_val}")
                continue

            # Branch on fractional variable
            branch_var = self._select_branching_variable(fractional_vars, current_node)

            # Create child nodes
            children = self._create_child_nodes(current_node, branch_var, obj_val, fractional_vars)

            # Add children to queue based on strategy
            self._add_nodes_to_queue(node_queue, children)

        # Phase 3: Final intensification (if using advanced heuristics)
        if (use_heuristic and self.heuristic_strategy == HeuristicStrategy.ADVANCED and
                self.adaptive_manager and self.best_solution_value < float('inf')):
            remaining_time = time_limit - (time.time() - start_time)
            final_phase_budget = time_limit * self.time_allocations['final']
            final_time = min(remaining_time, final_phase_budget)

            if final_time > 5.0:
                print(f"Starting final intensification phase with {final_time:.1f}s budget")
                current_best_sol = HeuristicSolution(
                    self.best_solution, self.best_solution_value, "branch_and_bound", 0
                )
                final_solution = self.adaptive_manager.phase3_final_intensification(current_best_sol)
                if final_solution and final_solution.cost < self.best_solution_value:
                    self.best_solution_value = final_solution.cost
                    self.best_solution = final_solution.solution
                    print(f"Final intensification improved solution to: {final_solution.cost}")

        solve_time = time.time() - start_time

        # Prepare heuristic info for result
        if self.best_solution_value < float('inf') and best_lower_bound > float('-inf'):
            gap = (self.best_solution_value - best_lower_bound) / max(abs(self.best_solution_value), 1e-10)
        else:
            gap = float('inf')
        is_optimal = (gap <= self.GAP_TOLERANCE) if gap < float('inf') else False

        heuristic_info = {
            'strategy': self.heuristic_strategy.value,
            'total_heuristic_time': self.total_heuristic_time,
            'heuristic_calls': self.heuristic_calls,
            'heuristic_improvements': self.heuristic_improvements,
            'performance': self.heuristic_performance,
            'gap': gap,
            'is_optimal': is_optimal,
            'best_lower_bound': best_lower_bound
        }

        return Result(self.best_solution_value, self.best_solution, solve_time,
                      self.nodes_explored, heuristic_info)

    def _run_initial_heuristics(self, time_budget):
        """Run initial heuristic phase"""
        start_time = time.time()
        solutions = []

        print(f"Running initial heuristics (budget: {time_budget:.1f}s)")

        if self.heuristic_strategy == HeuristicStrategy.BASIC:
            # Only constructive heuristics
            methods = [
                ('greedy_cost_effectiveness', self.constructive_heuristics.greedy_cost_effectiveness),
                ('greedy_max_coverage', self.constructive_heuristics.greedy_max_coverage),
                ('randomized_greedy', lambda: self.constructive_heuristics.randomized_greedy(iterations=3))
            ]

        elif self.heuristic_strategy == HeuristicStrategy.INTERMEDIATE:
            # Constructive + quick local search
            methods = [
                ('greedy_cost_effectiveness', self.constructive_heuristics.greedy_cost_effectiveness),
                ('greedy_max_coverage', self.constructive_heuristics.greedy_max_coverage),
                ('randomized_greedy', lambda: self.constructive_heuristics.randomized_greedy(iterations=5))
            ]

        elif self.heuristic_strategy == HeuristicStrategy.ADVANCED:
            # Use adaptive manager
            try:
                return self.adaptive_manager.phase1_quick_heuristics()
            except Exception as e:
                print(f"Advanced heuristic phase1 failed: {e}")
                return []

        elif self.heuristic_strategy == HeuristicStrategy.CUSTOM:
            # Custom mix
            methods = [
                ('greedy_cost_effectiveness', self.constructive_heuristics.greedy_cost_effectiveness),
                ('greedy_min_cost', self.constructive_heuristics.greedy_min_cost),
                ('randomized_greedy', lambda: self.constructive_heuristics.randomized_greedy(iterations=5))
            ]

        # Run the methods
        for method_name, method_func in methods:
            if time.time() - start_time > time_budget * 0.8:
                break

            try:
                heuristic_start = time.time()
                solution = method_func()
                heuristic_time = time.time() - heuristic_start

                solutions.append(solution)
                self.heuristic_calls += 1
                self.total_heuristic_time += heuristic_time

                # Track performance
                self.heuristic_performance['initial_phase'][method_name] = {
                    'cost': solution.cost,
                    'time': heuristic_time
                }
                self.heuristic_performance['method_calls'][method_name] = \
                    self.heuristic_performance['method_calls'].get(method_name, 0) + 1

                print(f"  {method_name}: cost={solution.cost:.1f}, time={heuristic_time:.3f}s")

            except Exception as e:
                print(f"  {method_name} failed: {e}")

        # Quick local search on best solution if we have intermediate/custom strategy
        if (solutions and self.heuristic_strategy in [HeuristicStrategy.INTERMEDIATE, HeuristicStrategy.CUSTOM]
                and time.time() - start_time < time_budget * 0.9):

            best_sol = min(solutions, key=lambda x: x.cost)
            try:
                heuristic_start = time.time()
                improved = self.local_search_heuristics.local_search_swap(best_sol, max_iterations=30)
                heuristic_time = time.time() - heuristic_start

                solutions.append(improved)
                self.total_heuristic_time += heuristic_time
                self.heuristic_calls += 1

                if improved.cost < best_sol.cost:
                    self.heuristic_improvements += 1

                print(f"  local_search_swap: cost={improved.cost:.1f}, time={heuristic_time:.3f}s")

            except Exception as e:
                print(f"  local_search_swap failed: {e}")

        return solutions

    def _should_run_periodic_heuristic(self, elapsed_time):
        """Determine if we should run periodic heuristic improvement"""
        # Don't run too frequently
        if self.nodes_since_last_heuristic < 100:
            return False

        initial_phase_time = self.current_time_limit * self.time_allocations['initial']

        # Don't run during initial phase
        if elapsed_time < initial_phase_time:
            return False

        # Run every 200 nodes or every 60 seconds
        if (self.nodes_since_last_heuristic >= 200 or
                elapsed_time - self.last_heuristic_time > 60):
            return True

        return False

    def _run_periodic_heuristic_improvement(self, elapsed_time):
        """Run periodic heuristic improvement during B&B"""
        if self.best_solution_value == float('inf'):
            return

        current_best = HeuristicSolution(
            self.best_solution, self.best_solution_value, "current_best", 0
        )

        improvement_found = False
        heuristic_start = time.time()

        try:
            if self.heuristic_strategy == HeuristicStrategy.INTERMEDIATE:
                # Quick local search
                improved = self.local_search_heuristics.local_search_drop_add(
                    current_best, max_iterations=20
                )

            elif self.heuristic_strategy == HeuristicStrategy.ADVANCED:
                # Use adaptive manager
                improved = self.adaptive_manager.periodic_heuristic_improvement(
                    current_best, time_budget=3.0
                )

            elif self.heuristic_strategy == HeuristicStrategy.CUSTOM:
                # Custom improvement
                improved = self.local_search_heuristics.local_search_swap(
                    current_best, max_iterations=30
                )
            else:
                improved = None

            if improved and improved.cost < self.best_solution_value:
                self.best_solution_value = improved.cost
                self.best_solution = improved.solution
                improvement_found = True
                self.heuristic_improvements += 1
                print(f"Periodic heuristic improvement: {improved.cost:.1f}")

        except Exception as e:
            print(f"Periodic heuristic failed: {e}")

        heuristic_time = time.time() - heuristic_start
        self.total_heuristic_time += heuristic_time
        self.heuristic_calls += 1
        self.nodes_since_last_heuristic = 0
        self.last_heuristic_time = elapsed_time

        # Track performance
        self.heuristic_performance['periodic_improvements'][f'call_{self.heuristic_calls}'] = {
            'improvement_found': improvement_found,
            'time': heuristic_time,
            'nodes_since_last': self.nodes_since_last_heuristic
        }

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

    def _select_branching_variable(self, fractional_vars, node):
        """Select which fractional variable to branch on"""
        # Enhanced branching variable selection using heuristic guidance
        if (self.heuristic_strategy == HeuristicStrategy.ADVANCED and
                self.adaptive_manager and len(fractional_vars) > 1):
            try:
                branch_var, method = self.adaptive_manager.get_branching_guidance(
                    fractional_vars, node.lower_bound
                )
                return branch_var
            except Exception as e:
                print(f"Advanced branching guidance failed: {e}")
                pass  # Fall back to default

        # Default strategy based on search strategy
        if self.strategy == SearchStrategy.MOST_FRACTIONAL:
            # Branch on variable closest to 0.5
            branch_var, _ = min(fractional_vars, key=lambda item: abs(0.5 - item[1]))
            return branch_var
        elif self.strategy == SearchStrategy.BEST_BOUND:
            best_var = None
            best_cost = -1
            for var_id, frac_val in fractional_vars:
                for atm in self.atm_list:
                    if atm.id == var_id:
                        if atm.cost > best_cost:
                            best_cost = atm.cost
                            best_var = var_id
                        break
            return best_var if best_var is not None else fractional_vars[0][0]
        else:
            # Default: branch on first fractional variable (simple strategy)
            return fractional_vars[0][0]

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

    def set_time_allocations(self, initial=0.40, main=0.50, final=0.10):
        """Set custom time allocations for different phases"""
        if abs(initial + main + final - 1.0) > 1e-6:
            raise ValueError("Time allocations must sum to 1.0")

        self.time_allocations = {
            'initial': initial,
            'main': main,
            'final': final
        }
        print(f"Updated time allocations: Initial={initial:.1%}, Main={main:.1%}, Final={final:.1%}")


def compare_strategies(filename, strategies=None, heuristic_strategies=None, runs=5, time_limit=60):
    """Compare different search strategies and heuristic combinations"""
    if strategies is None:
        strategies = [SearchStrategy.DEPTH_FIRST, SearchStrategy.BREADTH_FIRST,
                      SearchStrategy.BEST_FIRST, SearchStrategy.HYBRID_DF_BF]

    if heuristic_strategies is None:
        heuristic_strategies = [HeuristicStrategy.NONE, HeuristicStrategy.BASIC,
                                HeuristicStrategy.INTERMEDIATE]

    results = {}

    for strategy in strategies:
        for heuristic_strategy in heuristic_strategies:
            combo_name = f"{strategy.value}_{heuristic_strategy.value}"
            print(f"\nTesting combination: {combo_name}")

            times = []
            costs = []
            nodes = []
            heuristic_times = []
            heuristic_calls = []

            for run in range(runs):
                solver = BranchAndBoundSCPPriority(filename, strategy, heuristic_strategy)
                result = solver.solve(use_heuristic=(heuristic_strategy != HeuristicStrategy.NONE),
                                      time_limit=time_limit)

                times.append(result.getTime())
                costs.append(result.getOV())
                nodes.append(result.getNodesExplored())

                heuristic_info = result.getHeuristicInfo()
                heuristic_times.append(heuristic_info.get('total_heuristic_time', 0))
                heuristic_calls.append(heuristic_info.get('heuristic_calls', 0))

                print(f"  Run {run + 1}: Time={result.getTime():.4f}s, Cost={result.getOV()}, "
                      f"Nodes={result.getNodesExplored()}, HeurTime={heuristic_info.get('total_heuristic_time', 0):.3f}s, "
                      f"HeurCalls={heuristic_info.get('heuristic_calls', 0)}")

            results[combo_name] = {
                'strategy': strategy.value,
                'heuristic_strategy': heuristic_strategy.value,
                'avg_time': np.mean(times),
                'avg_cost': np.mean(costs),
                'avg_nodes': np.mean(nodes),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'std_time': np.std(times),
                'best_cost': np.min(costs),
                'avg_heuristic_time': np.mean(heuristic_times),
                'avg_heuristic_calls': np.mean(heuristic_calls)
            }

    # Print comparison
    print("\n" + "=" * 100)
    print("STRATEGY + HEURISTIC COMBINATION RESULTS")
    print("=" * 100)
    print(
        f"{'Combination':<30} {'Avg Time':<10} {'Best Cost':<10} {'Avg Nodes':<12} {'Heur Time':<10} {'Heur Calls':<10}")
    print("-" * 100)

    for combo_name, stats in results.items():
        print(f"{combo_name:<30} {stats['avg_time']:<10.4f} {stats['best_cost']:<10.0f} "
              f"{stats['avg_nodes']:<12.0f} {stats['avg_heuristic_time']:<10.3f} {stats['avg_heuristic_calls']:<10.0f}")

    return results


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # Test single strategy with heuristics
    print("Testing single strategy with heuristics...")
    solver = BranchAndBoundSCPPriority('scp44.txt', SearchStrategy.DEPTH_FIRST, HeuristicStrategy.INTERMEDIATE)
    result = solver.solve(use_heuristic=True, time_limit=30)
    print(f"Result: Cost={result.getOV()}, Time={result.getTime():.4f}s, Nodes={result.getNodesExplored()}")
    print(f"Heuristic Info: {result.getHeuristicInfo()}")

    # Compare strategies with heuristics
    print("\nComparing strategies with heuristics...")
    compare_results = compare_strategies('scp44.txt', runs=2, time_limit=30)
