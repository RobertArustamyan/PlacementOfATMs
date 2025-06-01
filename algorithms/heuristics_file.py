import random
import time
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import heapq
from collections import defaultdict


@dataclass
class HeuristicSolution:
    """Container for heuristic solutions"""
    solution: Dict[int, int]
    cost: float
    method: str
    time_taken: float
    
    def __lt__(self, other):
        return self.cost < other.cost


class ProblemAnalyzer:
    """Analyzes problem characteristics to guide heuristic selection"""
    
    def __init__(self, atm_list, user_coverage):
        self.atm_list = atm_list
        self.user_coverage = user_coverage
        self.num_sets = len(atm_list)
        self.num_users = len(user_coverage)
        self.analysis = self._analyze_problem()
    
    def _analyze_problem(self):
        """Analyze problem structure"""
        # Coverage analysis
        coverage_counts = [len(atm.covered_users_ids) for atm in self.atm_list]
        cost_ratios = [atm.cost / max(1, len(atm.covered_users_ids)) for atm in self.atm_list]
        
        # User coverage analysis
        user_coverage_counts = [len(sets) for sets in self.user_coverage.values()]
        
        return {
            'size_category': self._get_size_category(),
            'avg_coverage': np.mean(coverage_counts),
            'max_coverage': max(coverage_counts),
            'min_coverage': min(coverage_counts),
            'avg_cost_ratio': np.mean(cost_ratios),
            'coverage_variance': np.var(coverage_counts),
            'avg_user_coverage': np.mean(user_coverage_counts),
            'density': sum(coverage_counts) / (self.num_sets * self.num_users),
            'is_sparse': sum(coverage_counts) / (self.num_sets * self.num_users) < 0.1
        }
    
    def _get_size_category(self):
        """Categorize problem size"""
        if self.num_sets <= 50:
            return 'small'
        elif self.num_sets <= 200:
            return 'medium'
        else:
            return 'large'
    
    def get_recommended_strategies(self):
        """Get recommended heuristic strategies based on problem analysis"""
        strategies = []
        
        if self.analysis['size_category'] == 'small':
            strategies = ['greedy_variants', 'local_search', 'genetic_light']
        elif self.analysis['size_category'] == 'medium':
            strategies = ['greedy_variants', 'local_search', 'genetic_medium', 'simulated_annealing']
        else:
            strategies = ['greedy_variants', 'randomized_greedy', 'genetic_medium', 'lagrangian_heuristic']
        
        # Adjust based on density
        if self.analysis['is_sparse']:
            strategies.append('coverage_focused')
        
        return strategies


class ConstructiveHeuristics:
    """Collection of constructive heuristics for initial solutions"""
    
    def __init__(self, atm_list, user_coverage):
        self.atm_list = atm_list
        self.user_coverage = user_coverage
        self.atm_dict = {atm.id: atm for atm in atm_list}
    
    def greedy_cost_effectiveness(self) -> HeuristicSolution:
        """Standard greedy heuristic - best coverage per cost"""
        start_time = time.time()
        uncovered = set(self.user_coverage.keys())
        selected = set()
        total_cost = 0
        
        while uncovered:
            best_ratio = -1
            best_atm = None
            
            for atm in self.atm_list:
                if atm.id in selected:
                    continue
                
                new_coverage = len(uncovered.intersection(atm.covered_users_ids))
                if new_coverage == 0:
                    continue
                
                ratio = new_coverage / atm.cost
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_atm = atm
            
            if best_atm is None:
                break
            
            selected.add(best_atm.id)
            uncovered -= set(best_atm.covered_users_ids)
            total_cost += best_atm.cost
        
        solution = {atm.id: 1 if atm.id in selected else 0 for atm in self.atm_list}
        return HeuristicSolution(solution, total_cost, "greedy_cost_effectiveness", time.time() - start_time)
    
    def greedy_min_cost(self) -> HeuristicSolution:
        """Greedy heuristic - minimum cost first"""
        start_time = time.time()
        uncovered = set(self.user_coverage.keys())
        selected = set()
        total_cost = 0
        
        # Sort ATMs by cost
        sorted_atms = sorted(self.atm_list, key=lambda x: x.cost)
        
        while uncovered:
            best_atm = None
            best_coverage = 0
            
            for atm in sorted_atms:
                if atm.id in selected:
                    continue
                
                new_coverage = len(uncovered.intersection(atm.covered_users_ids))
                if new_coverage > best_coverage:
                    best_coverage = new_coverage
                    best_atm = atm
            
            if best_atm is None or best_coverage == 0:
                break
            
            selected.add(best_atm.id)
            uncovered -= set(best_atm.covered_users_ids)
            total_cost += best_atm.cost
        
        solution = {atm.id: 1 if atm.id in selected else 0 for atm in self.atm_list}
        return HeuristicSolution(solution, total_cost, "greedy_min_cost", time.time() - start_time)
    
    def greedy_max_coverage(self) -> HeuristicSolution:
        """Greedy heuristic - maximum coverage first"""
        start_time = time.time()
        uncovered = set(self.user_coverage.keys())
        selected = set()
        total_cost = 0
        
        while uncovered:
            best_coverage = 0
            best_atm = None
            
            for atm in self.atm_list:
                if atm.id in selected:
                    continue
                
                new_coverage = len(uncovered.intersection(atm.covered_users_ids))
                if new_coverage > best_coverage:
                    best_coverage = new_coverage
                    best_atm = atm
            
            if best_atm is None or best_coverage == 0:
                break
            
            selected.add(best_atm.id)
            uncovered -= set(best_atm.covered_users_ids)
            total_cost += best_atm.cost
        
        solution = {atm.id: 1 if atm.id in selected else 0 for atm in self.atm_list}
        return HeuristicSolution(solution, total_cost, "greedy_max_coverage", time.time() - start_time)
    
    def randomized_greedy(self, alpha=0.3, iterations=5) -> HeuristicSolution:
        """Randomized greedy construction"""
        best_solution = None
        
        for _ in range(iterations):
            start_time = time.time()
            uncovered = set(self.user_coverage.keys())
            selected = set()
            total_cost = 0
            
            while uncovered:
                # Calculate ratios for all feasible ATMs
                candidates = []
                for atm in self.atm_list:
                    if atm.id in selected:
                        continue
                    
                    new_coverage = len(uncovered.intersection(atm.covered_users_ids))
                    if new_coverage > 0:
                        ratio = new_coverage / atm.cost
                        candidates.append((atm, ratio))
                
                if not candidates:
                    break
                
                # Create restricted candidate list (RCL)
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_ratio = candidates[0][1]
                worst_ratio = candidates[-1][1]
                threshold = worst_ratio + alpha * (best_ratio - worst_ratio)
                
                rcl = [atm for atm, ratio in candidates if ratio >= threshold]
                
                # Randomly select from RCL
                chosen_atm = random.choice(rcl)
                selected.add(chosen_atm.id)
                uncovered -= set(chosen_atm.covered_users_ids)
                total_cost += chosen_atm.cost
            
            solution = {atm.id: 1 if atm.id in selected else 0 for atm in self.atm_list}
            current_solution = HeuristicSolution(solution, total_cost, "randomized_greedy", time.time() - start_time)
            
            if best_solution is None or current_solution.cost < best_solution.cost:
                best_solution = current_solution
        
        return best_solution


class LocalSearchHeuristics:
    """Local search and improvement heuristics"""
    
    def __init__(self, atm_list, user_coverage):
        self.atm_list = atm_list
        self.user_coverage = user_coverage
        self.atm_dict = {atm.id: atm for atm in atm_list}
    
    def local_search_swap(self, initial_solution: HeuristicSolution, max_iterations=100) -> HeuristicSolution:
        """Local search with swap neighborhood"""
        start_time = time.time()
        current_sol = initial_solution.solution.copy()
        current_cost = initial_solution.cost
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Try swapping each pair of variables
            selected_atms = [atm_id for atm_id, val in current_sol.items() if val == 1]
            unselected_atms = [atm_id for atm_id, val in current_sol.items() if val == 0]
            
            for selected_id in selected_atms:
                for unselected_id in unselected_atms:
                    # Try swap
                    new_sol = current_sol.copy()
                    new_sol[selected_id] = 0
                    new_sol[unselected_id] = 1
                    
                    if self._is_feasible(new_sol):
                        new_cost = self._calculate_cost(new_sol)
                        if new_cost < current_cost:
                            current_sol = new_sol
                            current_cost = new_cost
                            improved = True
                            break
                
                if improved:
                    break
        
        return HeuristicSolution(current_sol, current_cost, "local_search_swap", time.time() - start_time)
    
    def local_search_drop_add(self, initial_solution: HeuristicSolution, max_iterations=50) -> HeuristicSolution:
        """Local search with drop-add moves"""
        start_time = time.time()
        current_sol = initial_solution.solution.copy()
        current_cost = initial_solution.cost
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Try dropping each selected ATM and see if we can find a better replacement
            selected_atms = [atm_id for atm_id, val in current_sol.items() if val == 1]
            
            for drop_id in selected_atms:
                # Create solution without this ATM
                temp_sol = current_sol.copy()
                temp_sol[drop_id] = 0
                
                # Check if still feasible after dropping
                if self._is_feasible(temp_sol):
                    # It's feasible, so we have a better solution
                    new_cost = self._calculate_cost(temp_sol)
                    if new_cost < current_cost:
                        current_sol = temp_sol
                        current_cost = new_cost
                        improved = True
                        break
                else:
                    # Need to add something to make it feasible
                    uncovered = self._get_uncovered_users(temp_sol)
                    
                    # Find best ATM to add
                    best_add = None
                    best_cost_increase = float('inf')
                    
                    for atm in self.atm_list:
                        if temp_sol[atm.id] == 1:
                            continue
                        
                        if len(set(atm.covered_users_ids).intersection(uncovered)) > 0:
                            cost_increase = atm.cost - self.atm_dict[drop_id].cost
                            if cost_increase < best_cost_increase:
                                best_cost_increase = cost_increase
                                best_add = atm.id
                    
                    if best_add and best_cost_increase < 0:
                        temp_sol[best_add] = 1
                        if self._is_feasible(temp_sol):
                            new_cost = self._calculate_cost(temp_sol)
                            if new_cost < current_cost:
                                current_sol = temp_sol
                                current_cost = new_cost
                                improved = True
                                break
        
        return HeuristicSolution(current_sol, current_cost, "local_search_drop_add", time.time() - start_time)
    
    def variable_neighborhood_search(self, initial_solution: HeuristicSolution, time_limit=5) -> HeuristicSolution:
        """Variable Neighborhood Search"""
        start_time = time.time()
        best_solution = initial_solution
        current_solution = initial_solution
        
        neighborhoods = ['swap', 'drop_add', '2opt']
        
        while time.time() - start_time < time_limit:
            for neighborhood in neighborhoods:
                if neighborhood == 'swap':
                    new_solution = self.local_search_swap(current_solution, max_iterations=20)
                elif neighborhood == 'drop_add':
                    new_solution = self.local_search_drop_add(current_solution, max_iterations=20)
                else:  # 2opt
                    new_solution = self._two_opt_search(current_solution, max_iterations=20)
                
                if new_solution.cost < best_solution.cost:
                    best_solution = new_solution
                    current_solution = new_solution
                    break  # Move to next iteration with new best solution
            else:
                # No improvement in any neighborhood, try random restart
                current_solution = self._random_restart(current_solution)
        
        best_solution.method = "variable_neighborhood_search"
        best_solution.time_taken = time.time() - start_time
        return best_solution
    
    def _two_opt_search(self, initial_solution: HeuristicSolution, max_iterations=20) -> HeuristicSolution:
        """2-opt style search for set cover"""
        start_time = time.time()
        current_sol = initial_solution.solution.copy()
        current_cost = initial_solution.cost
        
        for _ in range(max_iterations):
            selected_atms = [atm_id for atm_id, val in current_sol.items() if val == 1]
            if len(selected_atms) < 2:
                break
            
            # Try removing two ATMs and adding two different ones
            for i in range(len(selected_atms)):
                for j in range(i + 1, len(selected_atms)):
                    temp_sol = current_sol.copy()
                    temp_sol[selected_atms[i]] = 0
                    temp_sol[selected_atms[j]] = 0
                    
                    # Find best two ATMs to add
                    uncovered = self._get_uncovered_users(temp_sol)
                    if not uncovered:
                        # Already feasible with removal
                        new_cost = self._calculate_cost(temp_sol)
                        if new_cost < current_cost:
                            current_sol = temp_sol
                            current_cost = new_cost
                            break
                    else:
                        # Need to add ATMs to cover uncovered users
                        self._greedy_repair(temp_sol, uncovered)
                        if self._is_feasible(temp_sol):
                            new_cost = self._calculate_cost(temp_sol)
                            if new_cost < current_cost:
                                current_sol = temp_sol
                                current_cost = new_cost
                                break
        
        return HeuristicSolution(current_sol, current_cost, "2opt_search", time.time() - start_time)
    
    def _random_restart(self, current_solution: HeuristicSolution) -> HeuristicSolution:
        """Create a random restart solution"""
        # Randomly remove 20% of selected ATMs and repair
        temp_sol = current_solution.solution.copy()
        selected_atms = [atm_id for atm_id, val in temp_sol.items() if val == 1]
        
        # Remove random subset
        to_remove = random.sample(selected_atms, max(1, len(selected_atms) // 5))
        for atm_id in to_remove:
            temp_sol[atm_id] = 0
        
        # Repair solution
        uncovered = self._get_uncovered_users(temp_sol)
        self._greedy_repair(temp_sol, uncovered)
        
        cost = self._calculate_cost(temp_sol)
        return HeuristicSolution(temp_sol, cost, "random_restart", 0)
    
    def _greedy_repair(self, solution: Dict[int, int], uncovered: Set[int]):
        """Greedily repair infeasible solution"""
        while uncovered:
            best_atm = None
            best_ratio = -1
            
            for atm in self.atm_list:
                if solution[atm.id] == 1:
                    continue
                
                new_coverage = len(uncovered.intersection(atm.covered_users_ids))
                if new_coverage > 0:
                    ratio = new_coverage / atm.cost
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_atm = atm
            
            if best_atm is None:
                break
            
            solution[best_atm.id] = 1
            uncovered -= set(best_atm.covered_users_ids)
    
    def _is_feasible(self, solution: Dict[int, int]) -> bool:
        """Check if solution is feasible"""
        covered_users = set()
        for atm_id, val in solution.items():
            if val == 1:
                covered_users.update(self.atm_dict[atm_id].covered_users_ids)
        
        return len(covered_users) >= len(self.user_coverage)
    
    def _get_uncovered_users(self, solution: Dict[int, int]) -> Set[int]:
        """Get uncovered users in current solution"""
        covered_users = set()
        for atm_id, val in solution.items():
            if val == 1:
                covered_users.update(self.atm_dict[atm_id].covered_users_ids)
        
        return set(self.user_coverage.keys()) - covered_users
    
    def _calculate_cost(self, solution: Dict[int, int]) -> float:
        """Calculate total cost of solution"""
        return sum(self.atm_dict[atm_id].cost for atm_id, val in solution.items() if val == 1)


class MetaHeuristics:
    """Quick metaheuristic algorithms"""
    
    def __init__(self, atm_list, user_coverage):
        self.atm_list = atm_list
        self.user_coverage = user_coverage
        self.atm_dict = {atm.id: atm for atm in atm_list}
    
    def genetic_algorithm(self, time_limit=10, population_size=20) -> HeuristicSolution:
        """Quick genetic algorithm"""
        start_time = time.time()
        
        # Initialize population with different heuristics
        constructive = ConstructiveHeuristics(self.atm_list, self.user_coverage)
        population = []
        
        # Create initial population
        for _ in range(population_size // 3):
            population.append(constructive.greedy_cost_effectiveness())
        for _ in range(population_size // 3):
            population.append(constructive.randomized_greedy())
        for _ in range(population_size - 2 * (population_size // 3)):
            population.append(constructive.greedy_max_coverage())
        
        best_solution = min(population, key=lambda x: x.cost)
        generation = 0
        
        while time.time() - start_time < time_limit:
            generation += 1
            new_population = []
            
            # Keep best solutions (elitism)
            population.sort(key=lambda x: x.cost)
            new_population.extend(population[:population_size // 4])
            
            # Generate offspring
            while len(new_population) < population_size:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                child = self._crossover(parent1, parent2)
                if random.random() < 0.1:  # 10% mutation rate
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
            current_best = min(population, key=lambda x: x.cost)
            
            if current_best.cost < best_solution.cost:
                best_solution = current_best
        
        best_solution.method = "genetic_algorithm"
        best_solution.time_taken = time.time() - start_time
        return best_solution
    
    def simulated_annealing(self, initial_solution: HeuristicSolution, time_limit=8) -> HeuristicSolution:
        """Simulated annealing algorithm"""
        start_time = time.time()
        current_solution = initial_solution
        best_solution = initial_solution
        
        # SA parameters
        initial_temp = 1000
        cooling_rate = 0.95
        temperature = initial_temp
        
        iteration = 0
        while time.time() - start_time < time_limit and temperature > 1:
            iteration += 1
            
            # Generate neighbor solution
            neighbor = self._generate_neighbor(current_solution)
            
            delta = neighbor.cost - current_solution.cost
            
            # Accept or reject
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                current_solution = neighbor
                
                if neighbor.cost < best_solution.cost:
                    best_solution = neighbor
            
            # Cool down
            if iteration % 10 == 0:
                temperature *= cooling_rate
        
        best_solution.method = "simulated_annealing"
        best_solution.time_taken = time.time() - start_time
        return best_solution
    
    def _tournament_selection(self, population, tournament_size=3):
        """Tournament selection for GA"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return min(tournament, key=lambda x: x.cost)
    
    def _crossover(self, parent1: HeuristicSolution, parent2: HeuristicSolution) -> HeuristicSolution:
        """Crossover operation for GA"""
        child_solution = {}
        
        # Use voting mechanism for crossover
        for atm_id in parent1.solution:
            vote1 = parent1.solution[atm_id]
            vote2 = parent2.solution[atm_id]
            
            if vote1 == vote2:
                child_solution[atm_id] = vote1
            else:
                # Random choice when parents disagree
                child_solution[atm_id] = random.choice([vote1, vote2])
        
        # Repair if infeasible
        if not self._is_feasible(child_solution):
            uncovered = self._get_uncovered_users(child_solution)
            self._greedy_repair(child_solution, uncovered)
        
        cost = self._calculate_cost(child_solution)
        return HeuristicSolution(child_solution, cost, "crossover", 0)
    
    def _mutate(self, solution: HeuristicSolution) -> HeuristicSolution:
        """Mutation operation for GA"""
        mutated_sol = solution.solution.copy()
        
        # Flip random bits
        atm_ids = list(mutated_sol.keys())
        num_mutations = max(1, len(atm_ids) // 10)
        
        for _ in range(num_mutations):
            atm_id = random.choice(atm_ids)
            mutated_sol[atm_id] = 1 - mutated_sol[atm_id]
        
        # Repair if infeasible
        if not self._is_feasible(mutated_sol):
            uncovered = self._get_uncovered_users(mutated_sol)
            self._greedy_repair(mutated_sol, uncovered)
        
        cost = self._calculate_cost(mutated_sol)
        return HeuristicSolution(mutated_sol, cost, "mutation", 0)
    
    def _generate_neighbor(self, solution: HeuristicSolution) -> HeuristicSolution:
        """Generate neighbor for SA"""
        neighbor_sol = solution.solution.copy()
        
        # Choose random move type
        move_type = random.choice(['flip', 'swap', 'shift'])
        
        if move_type == 'flip':
            # Flip one variable
            atm_id = random.choice(list(neighbor_sol.keys()))
            neighbor_sol[atm_id] = 1 - neighbor_sol[atm_id]
        
        elif move_type == 'swap':
            # Swap two variables
            selected = [aid for aid, val in neighbor_sol.items() if val == 1]
            unselected = [aid for aid, val in neighbor_sol.items() if val == 0]
            
            if selected and unselected:
                s1 = random.choice(selected)
                s2 = random.choice(unselected)
                neighbor_sol[s1] = 0
                neighbor_sol[s2] = 1
        
        else:  # shift
            # Remove one, add one
            selected = [aid for aid, val in neighbor_sol.items() if val == 1]
            if len(selected) > 1:
                to_remove = random.choice(selected)
                neighbor_sol[to_remove] = 0
                
                # Add random unselected
                unselected = [aid for aid, val in neighbor_sol.items() if val == 0]
                if unselected:
                    to_add = random.choice(unselected)
                    neighbor_sol[to_add] = 1
        
        # Repair if infeasible
        if not self._is_feasible(neighbor_sol):
            uncovered = self._get_uncovered_users(neighbor_sol)
            self._greedy_repair(neighbor_sol, uncovered)
        
        cost = self._calculate_cost(neighbor_sol)
        return HeuristicSolution(neighbor_sol, cost, "neighbor", 0)
    
    # Helper methods (same as in LocalSearchHeuristics)
    def _is_feasible(self, solution: Dict[int, int]) -> bool:
        covered_users = set()
        for atm_id, val in solution.items():
            if val == 1:
                covered_users.update(self.atm_dict[atm_id].covered_users_ids)
        return len(covered_users) >= len(self.user_coverage)
    
    def _get_uncovered_users(self, solution: Dict[int, int]) -> Set[int]:
        covered_users = set()
        for atm_id, val in solution.items():
            if val == 1:
                covered_users.update(self.atm_dict[atm_id].covered_users_ids)
        return set(self.user_coverage.keys()) - covered_users
    
    def _greedy_repair(self, solution: Dict[int, int], uncovered: Set[int]):
        while uncovered:
            best_atm = None
            best_ratio = -1
            
            for atm in self.atm_list:
                if solution[atm.id] == 1:
                    continue
                
                new_coverage = len(uncovered.intersection(atm.covered_users_ids))
                if new_coverage > 0:
                    ratio = new_coverage / atm.cost
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_atm = atm
            
            if best_atm is None:
                break
            
            solution[best_atm.id] = 1
            uncovered -= set(best_atm.covered_users_ids)
    
    def _calculate_cost(self, solution: Dict[int, int]) -> float:
        return sum(self.atm_dict[atm_id].cost for atm_id, val in solution.items() if val == 1)


class BoundingHeuristics:
    """Heuristics for bound computation and node pruning"""
    
    def __init__(self, atm_list, user_coverage):
        self.atm_list = atm_list
        self.user_coverage = user_coverage
        self.atm_dict = {atm.id: atm for atm in atm_list}

    def lagrangian_bound(self, multipliers=None, max_iterations=50) -> float:
        """Compute Lagrangian lower bound"""
        if multipliers is None:
            multipliers = {user_id: 1.0 for user_id in self.user_coverage.keys()}

        # Subgradient optimization for Lagrangian dual
        best_bound = float('-inf')
        step_size = 2.0

        for iteration in range(max_iterations):
            # Solve Lagrangian subproblem
            current_bound = 0
            solution = {}

            for atm in self.atm_list:
                # Calculate reduced cost
                reduced_cost = atm.cost
                for user_id in atm.covered_users_ids:
                    reduced_cost -= multipliers.get(user_id, 0)

                # Set variable value
                if reduced_cost < 0:
                    solution[atm.id] = 1
                    current_bound += reduced_cost
                else:
                    solution[atm.id] = 0

            # Add constraint violations (dual feasibility)
            for user_id in self.user_coverage.keys():
                coverage = sum(solution.get(atm_id, 0) for atm_id in self.user_coverage[user_id])
                if coverage >= 1:
                    current_bound += multipliers[user_id]
                else:
                    # Constraint is violated, penalize bound
                    current_bound += multipliers[user_id] * coverage

            # Update best bound
            if current_bound > best_bound:
                best_bound = current_bound

            # Compute subgradient
            subgradient = {}
            for user_id in self.user_coverage.keys():
                coverage = sum(solution.get(atm_id, 0) for atm_id in self.user_coverage[user_id])
                subgradient[user_id] = coverage - 1  # g_i = sum(a_ij * x_j) - 1

            # Check for optimality (subgradient close to zero)
            subgrad_norm = sum(g ** 2 for g in subgradient.values()) ** 0.5
            if subgrad_norm < 1e-6:
                break

            # Update multipliers using subgradient method
            # Step size adjustment for better convergence
            if iteration > 0 and iteration % 10 == 0:
                step_size *= 0.7  # Reduce step size over time

            for user_id in self.user_coverage.keys():
                multipliers[user_id] = max(0.0,
                                           multipliers[user_id] + step_size * subgradient[user_id] / subgrad_norm)

        return best_bound

    def linear_relaxation_bound(self, fixed_vars=None) -> Tuple[float, Dict[int, float]]:
        """Compute LP relaxation bound (simplified without Gurobi)"""
        if fixed_vars is None:
            fixed_vars = {}

        # Simple greedy fractional solution for bound estimation
        # This is a heuristic approximation of the LP bound
        remaining_users = set(self.user_coverage.keys())
        fractional_solution = {atm.id: 0.0 for atm in self.atm_list}
        total_cost = 0.0

        # Apply fixed variables
        for atm_id, value in fixed_vars.items():
            if value == 1:
                fractional_solution[atm_id] = 1.0
                atm = self.atm_dict[atm_id]
                total_cost += atm.cost
                remaining_users -= set(atm.covered_users_ids)
            elif value == 0:
                fractional_solution[atm_id] = 0.0

        # Greedy fractional assignment for remaining coverage
        iteration = 0
        while remaining_users and iteration < 100:
            iteration += 1
            best_ratio = 0
            best_atm = None

            for atm in self.atm_list:
                if fractional_solution[atm.id] >= 1.0:
                    continue

                # Calculate marginal coverage
                marginal_coverage = len(remaining_users.intersection(atm.covered_users_ids))
                if marginal_coverage == 0:
                    continue

                ratio = marginal_coverage / atm.cost
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_atm = atm

            if best_atm is None:
                break

            # Determine fractional value needed
            covered_users = set(best_atm.covered_users_ids).intersection(remaining_users)
            if covered_users:
                # Use minimum fractional value needed to cover at least one user
                fraction_needed = 1.0 / len(covered_users) if len(covered_users) > 1 else 1.0
                fraction_to_add = min(1.0 - fractional_solution[best_atm.id], fraction_needed)

                fractional_solution[best_atm.id] += fraction_to_add
                total_cost += best_atm.cost * fraction_to_add

                # Remove some users (proportional to fraction added)
                users_to_remove = list(covered_users)[:max(1, int(len(covered_users) * fraction_to_add))]
                remaining_users -= set(users_to_remove)

        return total_cost, fractional_solution

    def compute_lower_bound(self, method='lagrangian', **kwargs) -> float:
        """Unified interface for computing lower bounds"""
        if method == 'lagrangian':
            return self.lagrangian_bound(**kwargs)
        elif method == 'linear_relaxation':
            bound, _ = self.linear_relaxation_bound(**kwargs)
            return bound
        else:
            raise ValueError(f"Unknown bounding method: {method}")

    def pricing_heuristic(self, current_solution: Dict[int, int], dual_values: Dict[int, float] = None) -> List[int]:
        """Identify ATMs with negative reduced costs (good candidates to add)"""
        if dual_values is None:
            # Estimate dual values based on coverage gaps
            dual_values = {}
            covered_users = set()
            for atm_id, val in current_solution.items():
                if val == 1:
                    covered_users.update(self.atm_dict[atm_id].covered_users_ids)

            # Higher dual values for uncovered users
            for user_id in self.user_coverage.keys():
                dual_values[user_id] = 2.0 if user_id not in covered_users else 0.1

        candidates = []
        for atm in self.atm_list:
            if current_solution.get(atm.id, 0) == 1:
                continue

            # Calculate reduced cost
            reduced_cost = atm.cost
            for user_id in atm.covered_users_ids:
                reduced_cost -= dual_values.get(user_id, 0)

            if reduced_cost < -0.01:  # Negative reduced cost threshold
                candidates.append((atm.id, -reduced_cost))  # Store as positive improvement

        # Sort by improvement potential
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [atm_id for atm_id, _ in candidates[:5]]  # Return top 5 candidates


class AdaptiveManager:
    """Manages adaptive heuristic selection and execution strategies"""

    def __init__(self, atm_list, user_coverage, time_budget=300):
        self.atm_list = atm_list
        self.user_coverage = user_coverage
        self.time_budget = time_budget

        # Initialize components
        self.analyzer = ProblemAnalyzer(atm_list, user_coverage)
        self.constructive = ConstructiveHeuristics(atm_list, user_coverage)
        self.local_search = LocalSearchHeuristics(atm_list, user_coverage)
        self.meta_heuristics = MetaHeuristics(atm_list, user_coverage)
        self.bounding = BoundingHeuristics(atm_list, user_coverage)

        # Performance tracking
        self.method_performance = defaultdict(list)  # Track performance of each method
        self.solution_history = []
        self.current_best = None

        # Strategy state
        self.phase = 1
        self.last_improvement_time = 0
        self.stagnation_count = 0
        self.diversification_triggered = False

        # Timing allocations (as fractions of total budget)
        self.phase_allocations = {
            1: 0.10,  # Quick heuristic burst: 10%
            2: 0.80,  # Main B&B with periodic improvements: 80%
            3: 0.10  # Final intensification: 10%
        }

    def get_phase_time_budget(self, phase: int) -> float:
        """Get time budget for specific phase"""
        return self.time_budget * self.phase_allocations[phase]

    def phase1_quick_heuristics(self) -> List[HeuristicSolution]:
        """Phase 1: Quick heuristic burst to find good initial solutions"""
        start_time = time.time()
        phase_budget = self.get_phase_time_budget(1)
        solutions = []

        print(f"Phase 1: Quick heuristic burst ({phase_budget:.1f}s budget)")

        # 1. Fast constructive heuristics
        methods_to_try = [
            ('greedy_cost_effectiveness', self.constructive.greedy_cost_effectiveness),
            ('greedy_max_coverage', self.constructive.greedy_max_coverage),
            ('greedy_min_cost', self.constructive.greedy_min_cost)
        ]

        for method_name, method in methods_to_try:
            if time.time() - start_time > phase_budget * 0.4:
                break

            try:
                sol = method()
                solutions.append(sol)
                self._update_performance_tracking(method_name, sol)
                print(f"  {method_name}: cost={sol.cost:.1f}, time={sol.time_taken:.3f}s")
            except Exception as e:
                print(f"  {method_name} failed: {e}")

        # 2. Randomized greedy with multiple runs
        remaining_time = phase_budget * 0.4 - (time.time() - start_time)
        if remaining_time > 1.0:
            iterations = max(3, int(remaining_time / 0.5))  # ~0.5s per iteration
            try:
                sol = self.constructive.randomized_greedy(alpha=0.3, iterations=iterations)
                solutions.append(sol)
                self._update_performance_tracking('randomized_greedy', sol)
                print(f"  randomized_greedy: cost={sol.cost:.1f}, time={sol.time_taken:.3f}s")
            except Exception as e:
                print(f"  randomized_greedy failed: {e}")

        # 3. Quick local search on best solution so far
        if solutions:
            best_so_far = min(solutions, key=lambda x: x.cost)
            remaining_time = phase_budget - (time.time() - start_time)

            if remaining_time > 2.0:
                try:
                    improved = self.local_search.local_search_swap(best_so_far, max_iterations=50)
                    solutions.append(improved)
                    self._update_performance_tracking('local_search_swap', improved)
                    print(f"  local_search_swap: cost={improved.cost:.1f}, time={improved.time_taken:.3f}s")
                except Exception as e:
                    print(f"  local_search_swap failed: {e}")

        # Update current best
        if solutions:
            self.current_best = min(solutions, key=lambda x: x.cost)
            self.last_improvement_time = time.time()
            print(f"Phase 1 best: {self.current_best.cost:.1f} (method: {self.current_best.method})")

        return solutions

    def get_branching_guidance(self, fractional_vars: List[Tuple[int, float]],
                               current_bound: float) -> Tuple[int, str]:
        """Provide guidance for variable selection in branch-and-bound"""

        # Strategy 1: Most fractional (default)
        most_fractional = min(fractional_vars, key=lambda x: abs(0.5 - x[1]))

        # Strategy 2: Reduced cost based (if we have dual information)
        if hasattr(self, '_last_dual_values'):
            try:
                # Calculate reduced costs
                reduced_costs = []
                for var_id, frac_val in fractional_vars:
                    atm = next(atm for atm in self.atm_list if atm.id == var_id)
                    reduced_cost = atm.cost
                    for user_id in atm.covered_users_ids:
                        reduced_cost -= self._last_dual_values.get(user_id, 0)
                    reduced_costs.append((var_id, frac_val, abs(reduced_cost)))

                # Branch on variable with highest reduced cost magnitude
                best_reduced_cost = max(reduced_costs, key=lambda x: x[2])
                if best_reduced_cost[2] > 0.01:  # Significant reduced cost
                    return best_reduced_cost[0], "reduced_cost"
            except:
                pass  # Fall back to most fractional

        # Strategy 3: Coverage-based priority
        if len(fractional_vars) > 3:
            # Prioritize variables covering many uncovered users
            coverage_scores = []
            for var_id, frac_val in fractional_vars:
                atm = next(atm for atm in self.atm_list if atm.id == var_id)
                # Estimate uncovered users this ATM could help with
                coverage_score = len(atm.covered_users_ids) * (1 - abs(2 * frac_val - 1))
                coverage_scores.append((var_id, coverage_score))

            best_coverage = max(coverage_scores, key=lambda x: x[1])
            return best_coverage[0], "coverage_based"

        return most_fractional[0], "most_fractional"

    def should_call_heuristic_improvement(self, nodes_since_last: int,
                                          time_since_start: float,
                                          current_gap: float = None) -> bool:
        """Decide when to call heuristic improvement during B&B"""

        # Don't improve too frequently (computational overhead)
        if nodes_since_last < 50:
            return False

        # Time-based triggers
        if time_since_start > self.get_phase_time_budget(1):  # After phase 1
            # Call every 100 nodes or every 30 seconds
            if nodes_since_last >= 100 or time_since_start - self.last_improvement_time > 30:
                return True

        # Gap-based trigger (if gap information available)
        if current_gap is not None and current_gap < 0.05:  # Gap < 5%
            # More frequent improvements when close to optimal
            return nodes_since_last >= 25

        # Stagnation trigger
        if time_since_start - self.last_improvement_time > 60:  # No improvement for 1 minute
            self.stagnation_count += 1
            return True

        return False

    def periodic_heuristic_improvement(self, current_incumbent: HeuristicSolution,
                                       time_budget: float = 5.0) -> Optional[HeuristicSolution]:
        """Run heuristic improvement during B&B phase"""
        start_time = time.time()

        if current_incumbent is None:
            return None

        # Select improvement method based on performance history and time available
        if time_budget >= 8.0:
            # Long time budget: try metaheuristic
            methods = [
                ('variable_neighborhood_search',
                 lambda: self.local_search.variable_neighborhood_search(current_incumbent, time_budget - 1)),
                ('simulated_annealing',
                 lambda: self.meta_heuristics.simulated_annealing(current_incumbent, time_budget - 1))
            ]
        else:
            # Short time budget: quick local search
            methods = [
                ('local_search_drop_add',
                 lambda: self.local_search.local_search_drop_add(current_incumbent, max_iterations=30)),
                ('local_search_swap',
                 lambda: self.local_search.local_search_swap(current_incumbent, max_iterations=50))
            ]

        best_improvement = None

        for method_name, method_func in methods:
            if time.time() - start_time > time_budget * 0.9:
                break

            try:
                improved_sol = method_func()
                if improved_sol.cost < current_incumbent.cost:
                    if best_improvement is None or improved_sol.cost < best_improvement.cost:
                        best_improvement = improved_sol
                        self.last_improvement_time = time.time()
                        print(f"  Heuristic improvement: {improved_sol.cost:.1f} via {method_name}")

                self._update_performance_tracking(method_name, improved_sol)

            except Exception as e:
                print(f"  Heuristic {method_name} failed: {e}")

        return best_improvement

    def phase3_final_intensification(self, current_best: HeuristicSolution) -> HeuristicSolution:
        """Phase 3: Final intensification with remaining time"""
        start_time = time.time()
        phase_budget = self.get_phase_time_budget(3)

        print(f"Phase 3: Final intensification ({phase_budget:.1f}s budget)")

        if current_best is None:
            return current_best

        best_solution = current_best

        # 1. Intensive local search
        if phase_budget > 5.0:
            try:
                vns_solution = self.local_search.variable_neighborhood_search(
                    best_solution, time_limit=min(phase_budget * 0.4, 10.0))
                if vns_solution.cost < best_solution.cost:
                    best_solution = vns_solution
                    print(f"  VNS improvement: {vns_solution.cost:.1f}")
            except Exception as e:
                print(f"  VNS failed: {e}")

        # 2. Genetic algorithm for diversification
        remaining_time = phase_budget - (time.time() - start_time)
        if remaining_time > 8.0:
            try:
                ga_solution = self.meta_heuristics.genetic_algorithm(
                    time_limit=min(remaining_time * 0.6, 15.0))
                if ga_solution.cost < best_solution.cost:
                    best_solution = ga_solution
                    print(f"  GA improvement: {ga_solution.cost:.1f}")
            except Exception as e:
                print(f"  GA failed: {e}")

        # 3. Final local search on best found
        remaining_time = phase_budget - (time.time() - start_time)
        if remaining_time > 2.0 and best_solution != current_best:
            try:
                final_ls = self.local_search.local_search_swap(
                    best_solution, max_iterations=100)
                if final_ls.cost < best_solution.cost:
                    best_solution = final_ls
                    print(f"  Final LS improvement: {final_ls.cost:.1f}")
            except Exception as e:
                print(f"  Final LS failed: {e}")

        return best_solution

    def get_bound_computation_strategy(self, node_depth: int, time_remaining: float) -> str:
        """Choose bounding strategy based on context"""

        # Early in search: use fast bounds
        if node_depth <= 3 or time_remaining < 60:
            return 'linear_relaxation'

        # Later in search with sufficient time: use better bounds
        if time_remaining > 120 and node_depth <= 10:
            return 'lagrangian'

        # Default
        return 'linear_relaxation'

    def _update_performance_tracking(self, method_name: str, solution: HeuristicSolution):
        """Track performance of different methods"""
        self.method_performance[method_name].append({
            'cost': solution.cost,
            'time': solution.time_taken,
            'cost_per_time': solution.cost / max(solution.time_taken, 0.001)
        })

        # Keep only recent history (last 10 runs per method)
        if len(self.method_performance[method_name]) > 10:
            self.method_performance[method_name] = self.method_performance[method_name][-10:]

    def get_performance_summary(self) -> Dict:
        """Get summary of method performances"""
        summary = {}
        for method, performances in self.method_performance.items():
            if performances:
                costs = [p['cost'] for p in performances]
                times = [p['time'] for p in performances]
                summary[method] = {
                    'avg_cost': np.mean(costs),
                    'best_cost': min(costs),
                    'avg_time': np.mean(times),
                    'runs': len(performances)
                }
        return summary

    def should_terminate_early(self, current_gap: float, time_elapsed: float) -> bool:
        """Decide if we should terminate B&B early"""

        # Gap-based termination
        if current_gap is not None and current_gap < 0.001:  # 0.1% gap
            return True

        # Time-based with stagnation
        if time_elapsed > self.time_budget * 0.8:  # 80% of time used
            if time_elapsed - self.last_improvement_time > self.time_budget * 0.2:  # No improvement in 20% of budget
                return True

        return False