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
            
            # Add constraint violations
            for user_id in self.user_coverage.keys():
                coverage = sum(solution.get(