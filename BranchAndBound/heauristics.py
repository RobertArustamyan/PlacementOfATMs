import random
import time
import copy
import numpy as np
from utils.scp_data_reading import read_scp_raw_data, convert_scp_data_to_objects


class SCPMetaheuristics:
    """
    Enhanced metaheuristic approaches for the Set Covering Problem
    """

    def __init__(self, filename):
        """
        Initialize the metaheuristic solver.

        Args:
            filename: SCP problem file path
        """
        self.filename = filename

        # Read problem data
        self.set_costs, self.user_coverage = read_scp_raw_data(filename)
        self.atm_list = convert_scp_data_to_objects(self.set_costs, self.user_coverage)

        # Create reverse coverage map (which users are covered by each ATM)
        self.atm_coverage = {}
        for atm in self.atm_list:
            self.atm_coverage[atm.id] = atm.covered_users_ids

        # Create coverage matrix for quick lookup
        self.all_users = set(self.user_coverage.keys())
        self.all_atms = {atm.id for atm in self.atm_list}

        # Map for ATM costs
        self.atm_costs = {atm.id: atm.cost for atm in self.atm_list}

    def evaluate_solution(self, solution):
        """
        Evaluate a solution (cost and feasibility)

        Args:
            solution: Dictionary with ATM ids as keys and 0/1 as values

        Returns:
            tuple: (cost, is_feasible, uncovered_users)
        """
        selected_atms = {atm_id for atm_id, val in solution.items() if val == 1}
        covered_users = set()

        for atm_id in selected_atms:
            covered_users.update(self.atm_coverage[atm_id])

        cost = sum(self.atm_costs[atm_id] for atm_id in selected_atms)
        uncovered = self.all_users - covered_users
        is_feasible = len(uncovered) == 0

        return cost, is_feasible, uncovered

    def greedy_construction(self, randomized=False, alpha=0.3):
        """
        Greedy or Randomized Greedy (GRASP) construction

        Args:
            randomized: Use randomized selection from RCL
            alpha: Greediness factor (0=pure greedy, 1=random)

        Returns:
            dict: Constructed solution
        """
        uncovered_users = set(self.user_coverage.keys())
        solution = {atm.id: 0 for atm in self.atm_list}

        while uncovered_users:
            # Calculate score for each ATM
            scores = {}
            for atm in self.atm_list:
                if solution[atm.id] == 1:
                    continue  # Already selected

                covers = uncovered_users.intersection(atm.covered_users_ids)
                if not covers:
                    continue

                # Score: users covered per unit cost
                scores[atm.id] = len(covers) / atm.cost

            if not scores:
                # No ATM can cover remaining users
                break

            if randomized:
                # GRASP: construct Restricted Candidate List (RCL)
                min_score = min(scores.values())
                max_score = max(scores.values())
                threshold = min_score + alpha * (max_score - min_score)
                rcl = [atm_id for atm_id, score in scores.items() if score >= threshold]

                # Select randomly from RCL
                selected_atm = random.choice(rcl)
            else:
                # Pure greedy: select best score
                selected_atm = max(scores.items(), key=lambda x: x[1])[0]

            # Update solution and uncovered users
            solution[selected_atm] = 1
            uncovered_users -= set(self.atm_coverage[selected_atm])

        cost, is_feasible, _ = self.evaluate_solution(solution)

        if not is_feasible:
            print(
                f"Warning: {'Randomized ' if randomized else ''}Greedy construction could not find feasible solution!")

        return solution

    def local_search(self, solution, max_iterations=1000, strategy="first_improvement"):
        """
        Local search to improve solution

        Args:
            solution: Initial solution
            max_iterations: Maximum number of iterations
            strategy: "first_improvement" or "best_improvement"

        Returns:
            dict: Improved solution
        """
        best_solution = solution.copy()
        best_cost, is_feasible, _ = self.evaluate_solution(best_solution)

        if not is_feasible:
            print("Warning: Initial solution is not feasible for local search!")
            return best_solution

        iteration = 0
        improved = True

        while improved and iteration < max_iterations:
            improved = False
            current_solution = best_solution.copy()
            current_cost = best_cost

            # Try removing redundant ATMs
            selected_atms = [atm_id for atm_id, val in current_solution.items() if val == 1]

            # Shuffle for more randomization
            random.shuffle(selected_atms)

            for atm_id in selected_atms:
                # Try removing this ATM
                temp_solution = current_solution.copy()
                temp_solution[atm_id] = 0

                temp_cost, temp_feasible, _ = self.evaluate_solution(temp_solution)

                if temp_feasible and temp_cost < current_cost:
                    if strategy == "first_improvement":
                        current_solution = temp_solution
                        current_cost = temp_cost
                        improved = True
                        break
                    elif strategy == "best_improvement":
                        # Track best improvement
                        if temp_cost < best_cost:
                            best_solution = temp_solution
                            best_cost = temp_cost
                            improved = True

            # If we found improvement with first improvement strategy
            if strategy == "first_improvement" and improved:
                best_solution = current_solution
                best_cost = current_cost

            iteration += 1

        return best_solution

    def variable_neighborhood_descent(self, solution, max_iterations=100):
        """
        Variable Neighborhood Descent

        Args:
            solution: Initial solution
            max_iterations: Maximum iterations

        Returns:
            dict: Improved solution
        """
        best_solution = solution.copy()
        best_cost, is_feasible, _ = self.evaluate_solution(best_solution)

        if not is_feasible:
            return best_solution

        # Define neighborhood structures
        neighborhoods = [
            self._swap_neighborhood,
            self._flip_neighborhood,
            self._insert_remove_neighborhood
        ]

        iteration = 0
        k = 0  # Current neighborhood

        while k < len(neighborhoods) and iteration < max_iterations:
            # Get improved solution from current neighborhood
            new_solution = neighborhoods[k](best_solution)
            new_cost, new_feasible, _ = self.evaluate_solution(new_solution)

            if new_feasible and new_cost < best_cost:
                # Improvement found, reset to first neighborhood
                best_solution = new_solution
                best_cost = new_cost
                k = 0
            else:
                # No improvement, move to next neighborhood
                k += 1

            iteration += 1

        return best_solution

    def _swap_neighborhood(self, solution):
        """Swap neighborhood: swap 0 with 1"""
        best_solution = solution.copy()
        best_cost, best_feasible, _ = self.evaluate_solution(best_solution)

        selected = [atm_id for atm_id, val in solution.items() if val == 1]
        not_selected = [atm_id for atm_id, val in solution.items() if val == 0]

        for s in random.sample(selected, min(len(selected), 10)):  # Limit search for performance
            for n in random.sample(not_selected, min(len(not_selected), 10)):
                temp_solution = solution.copy()
                temp_solution[s] = 0
                temp_solution[n] = 1

                temp_cost, temp_feasible, _ = self.evaluate_solution(temp_solution)

                if temp_feasible and temp_cost < best_cost:
                    best_solution = temp_solution
                    best_cost = temp_cost

        return best_solution

    def _flip_neighborhood(self, solution):
        """Flip neighborhood: flip a binary value"""
        best_solution = solution.copy()
        best_cost, best_feasible, _ = self.evaluate_solution(best_solution)

        for atm_id in random.sample(list(solution.keys()), min(len(solution), 20)):  # Limit search
            temp_solution = solution.copy()
            temp_solution[atm_id] = 1 - temp_solution[atm_id]  # Flip 0->1 or 1->0

            temp_cost, temp_feasible, _ = self.evaluate_solution(temp_solution)

            if temp_feasible and temp_cost < best_cost:
                best_solution = temp_solution
                best_cost = temp_cost

        return best_solution

    def _insert_remove_neighborhood(self, solution):
        """Insert or remove neighborhood: add one ATM or remove one"""
        best_solution = solution.copy()
        best_cost, best_feasible, _ = self.evaluate_solution(best_solution)

        # Try removing one
        selected = [atm_id for atm_id, val in solution.items() if val == 1]
        for atm_id in random.sample(selected, min(len(selected), 10)):
            temp_solution = solution.copy()
            temp_solution[atm_id] = 0

            temp_cost, temp_feasible, _ = self.evaluate_solution(temp_solution)

            if temp_feasible and temp_cost < best_cost:
                best_solution = temp_solution
                best_cost = temp_cost

        # Try adding one
        not_selected = [atm_id for atm_id, val in solution.items() if val == 0]
        for atm_id in random.sample(not_selected, min(len(not_selected), 10)):
            temp_solution = solution.copy()
            temp_solution[atm_id] = 1

            temp_cost, temp_feasible, _ = self.evaluate_solution(temp_solution)

            if temp_feasible and temp_cost < best_cost:
                best_solution = temp_solution
                best_cost = temp_cost

        return best_solution

    def tabu_search(self, initial_solution=None, max_iterations=100, tabu_tenure=10):
        """
        Tabu Search metaheuristic

        Args:
            initial_solution: Starting solution (if None, will use greedy)
            max_iterations: Maximum number of iterations
            tabu_tenure: Length of tabu list

        Returns:
            dict: Best solution found
        """
        # Get initial solution if not provided
        if initial_solution is None:
            initial_solution = self.greedy_construction()

        current_solution = initial_solution.copy()
        best_solution = initial_solution.copy()

        current_cost, current_feasible, _ = self.evaluate_solution(current_solution)
        best_cost = current_cost

        # Initialize tabu list
        tabu_list = {}  # Dictionary with atm_id as key and tenure as value

        iteration = 0
        while iteration < max_iterations:
            # Generate neighbors by flipping one variable
            best_neighbor = None
            best_neighbor_cost = float('inf')

            # Try flipping each variable
            for atm_id in self.all_atms:
                # Skip if in tabu list (unless it gives the best solution so far - aspiration)
                if atm_id in tabu_list and tabu_list[atm_id] > 0:
                    continue

                # Flip the variable
                neighbor = current_solution.copy()
                neighbor[atm_id] = 1 - neighbor[atm_id]

                # Evaluate neighbor
                neighbor_cost, neighbor_feasible, _ = self.evaluate_solution(neighbor)

                # Accept if feasible and better than current best neighbor
                if neighbor_feasible and neighbor_cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost
                    flipped_atm = atm_id

            # If no feasible neighbor found
            if best_neighbor is None:
                break

            # Move to best neighbor
            current_solution = best_neighbor
            current_cost = best_neighbor_cost

            # Add to tabu list
            tabu_list[flipped_atm] = tabu_tenure

            # Update best solution if improved
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost

            # Decrease tabu tenure
            for atm_id in list(tabu_list.keys()):
                tabu_list[atm_id] -= 1
                if tabu_list[atm_id] <= 0:
                    del tabu_list[atm_id]

            iteration += 1

        return best_solution

    def simulated_annealing(self, initial_solution=None, initial_temp=100.0, cooling_rate=0.95,
                            min_temp=0.1, iterations_per_temp=10):
        """
        Simulated Annealing metaheuristic

        Args:
            initial_solution: Starting solution (if None, will use greedy)
            initial_temp: Starting temperature
            cooling_rate: Temperature cooling rate per iteration
            min_temp: Minimum temperature to stop
            iterations_per_temp: Number of iterations at each temperature

        Returns:
            dict: Best solution found
        """
        # Get initial solution if not provided
        if initial_solution is None:
            initial_solution = self.greedy_construction()

        current_solution = initial_solution.copy()
        best_solution = initial_solution.copy()

        current_cost, current_feasible, _ = self.evaluate_solution(current_solution)
        best_cost = current_cost

        if not current_feasible:
            print("Warning: Initial solution for SA is not feasible!")
            return best_solution

        temp = initial_temp

        while temp > min_temp:
            for i in range(iterations_per_temp):
                # Generate random neighbor
                neighbor = current_solution.copy()

                # Randomly select whether to add, remove, or swap
                move_type = random.choice(['add', 'remove', 'swap'])

                if move_type == 'add':
                    # Add one ATM
                    not_selected = [atm_id for atm_id, val in neighbor.items() if val == 0]
                    if not_selected:
                        atm_to_add = random.choice(not_selected)
                        neighbor[atm_to_add] = 1
                elif move_type == 'remove':
                    # Remove one ATM
                    selected = [atm_id for atm_id, val in neighbor.items() if val == 1]
                    if selected:
                        atm_to_remove = random.choice(selected)
                        neighbor[atm_to_remove] = 0
                else:  # swap
                    # Swap one ATM in for one out
                    selected = [atm_id for atm_id, val in neighbor.items() if val == 1]
                    not_selected = [atm_id for atm_id, val in neighbor.items() if val == 0]
                    if selected and not_selected:
                        atm_to_remove = random.choice(selected)
                        atm_to_add = random.choice(not_selected)
                        neighbor[atm_to_remove] = 0
                        neighbor[atm_to_add] = 1

                # Evaluate neighbor
                neighbor_cost, neighbor_feasible, _ = self.evaluate_solution(neighbor)

                # Skip infeasible solutions
                if not neighbor_feasible:
                    continue

                # Calculate acceptance probability
                delta = neighbor_cost - current_cost

                # If better, or passes probability check
                if delta < 0 or random.random() < np.exp(-delta / temp):
                    current_solution = neighbor
                    current_cost = neighbor_cost

                    # Update best solution if improved
                    if current_cost < best_cost:
                        best_solution = current_solution.copy()
                        best_cost = current_cost

            # Cool down temperature
            temp *= cooling_rate

        return best_solution

    def genetic_algorithm(self, population_size=50, generations=100, mutation_rate=0.1, crossover_rate=0.8):
        """
        Genetic Algorithm metaheuristic

        Args:
            population_size: Size of the population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover

        Returns:
            dict: Best solution found
        """
        # Initialize population
        population = []
        for _ in range(population_size):
            # Create individual with randomized greedy (GRASP)
            solution = self.greedy_construction(randomized=True, alpha=random.uniform(0.2, 0.8))
            cost, feasible, _ = self.evaluate_solution(solution)

            if feasible:
                population.append((solution, cost))

        # If population is too small (not enough feasible solutions)
        while len(population) < population_size:
            solution = self.repair_solution({atm.id: random.choice([0, 1]) for atm in self.atm_list})
            cost, feasible, _ = self.evaluate_solution(solution)
            if feasible:
                population.append((solution, cost))

        # Main GA loop
        for generation in range(generations):
            new_population = []

            # Elitism: keep best solutions
            elite_count = max(1, int(population_size * 0.1))
            elite = sorted(population, key=lambda x: x[1])[:elite_count]
            new_population.extend(elite)

            # Create rest of population
            while len(new_population) < population_size:
                # Selection
                p1 = self._tournament_selection(population)
                p2 = self._tournament_selection(population)

                # Crossover
                if random.random() < crossover_rate:
                    offspring = self._crossover(p1[0], p2[0])
                else:
                    offspring = p1[0].copy()

                # Mutation
                if random.random() < mutation_rate:
                    offspring = self._mutate(offspring)

                # Repair if needed
                offspring = self.repair_solution(offspring)

                # Add to new population
                cost, feasible, _ = self.evaluate_solution(offspring)
                if feasible:
                    new_population.append((offspring, cost))

            # Replace population
            population = new_population

        # Return best solution
        return min(population, key=lambda x: x[1])[0]

    def _tournament_selection(self, population, tournament_size=3):
        """Tournament selection for genetic algorithm"""
        tournament = random.sample(population, tournament_size)
        return min(tournament, key=lambda x: x[1])

    def _crossover(self, parent1, parent2):
        """One-point crossover for genetic algorithm"""
        offspring = {}
        cutpoint = random.randint(0, len(parent1) - 1)

        atm_ids = list(parent1.keys())

        for i, atm_id in enumerate(atm_ids):
            if i < cutpoint:
                offspring[atm_id] = parent1[atm_id]
            else:
                offspring[atm_id] = parent2[atm_id]

        return offspring

    def _mutate(self, solution):
        """Mutation for genetic algorithm - flip random bits"""
        mutated = solution.copy()

        # Flip a few random bits
        num_flips = max(1, int(len(solution) * 0.05))  # Flip about 5% of bits

        for _ in range(num_flips):
            atm_id = random.choice(list(solution.keys()))
            mutated[atm_id] = 1 - mutated[atm_id]  # Flip

        return mutated

    def repair_solution(self, solution):
        """Repair an infeasible solution"""
        repaired = solution.copy()
        cost, feasible, uncovered = self.evaluate_solution(repaired)

        if feasible:
            # Already feasible, try to remove redundant ATMs
            return self.local_search(repaired, max_iterations=10)

        # Add ATMs to cover uncovered users
        while uncovered:
            best_atm = None
            best_score = float('-inf')

            for atm in self.atm_list:
                if repaired[atm.id] == 1:
                    continue  # Already selected

                # Check how many uncovered users this ATM would cover
                covers = uncovered.intersection(atm.covered_users_ids)
                if not covers:
                    continue

                # Score: users covered per unit cost
                score = len(covers) / atm.cost

                if score > best_score:
                    best_score = score
                    best_atm = atm

            if best_atm is None:
                # No ATM can cover remaining users (shouldn't happen)
                break

            # Add ATM to solution
            repaired[best_atm.id] = 1
            uncovered -= set(best_atm.covered_users_ids)

        return repaired

    def hybrid_vns_sa(self, max_iterations=50):
        """
        Hybrid metaheuristic combining VNS and SA

        Args:
            max_iterations: Maximum iterations

        Returns:
            dict: Best solution found
        """
        # Start with greedy solution
        solution = self.greedy_construction()
        cost, feasible, _ = self.evaluate_solution(solution)

        if not feasible:
            solution = self.repair_solution(solution)

        # Apply VND to get good initial solution
        solution = self.variable_neighborhood_descent(solution, max_iterations=10)

        # Apply SA with reduced temperature
        solution = self.simulated_annealing(
            initial_solution=solution,
            initial_temp=50.0,
            cooling_rate=0.9,
            iterations_per_temp=5
        )

        return solution

    def run_all_metaheuristics(self, time_limit=10):
        """
        Run all metaheuristics and return the best solution

        Args:
            time_limit: Time limit in seconds

        Returns:
            dict: Best solution found
        """
        start_time = time.time()
        best_solution = None
        best_cost = float('inf')

        # List of all metaheuristics to try
        methods = [
            ("Greedy", lambda: self.greedy_construction()),
            ("GRASP", lambda: self.greedy_construction(randomized=True)),
            ("Local Search", lambda: self.local_search(self.greedy_construction(), max_iterations=100)),
            ("VND", lambda: self.variable_neighborhood_descent(self.greedy_construction(), max_iterations=20)),
            ("Tabu Search", lambda: self.tabu_search(max_iterations=20)),
            ("Simulated Annealing", lambda: self.simulated_annealing()),
            ("Genetic Algorithm", lambda: self.genetic_algorithm(population_size=20, generations=20)),
            ("Hybrid VNS-SA", lambda: self.hybrid_vns_sa())
        ]

        results = []

        for name, method in methods:
            if time.time() - start_time > time_limit:
                print("Time limit reached")
                break

            method_start = time.time()
            solution = method()
            method_time = time.time() - method_start

            cost, feasible, _ = self.evaluate_solution(solution)

            if feasible and cost < best_cost:
                best_solution = solution
                best_cost = cost

            results.append({
                "Method": name,
                "Cost": cost,
                "Feasible": feasible,
                "Time": method_time
            })

            # print(f"{name}: Cost={cost}, Feasible={feasible}, Time={method_time:.4f}s")

        # Print summary
        # print("\n===== Metaheuristics Results =====")
        # print(f"{'Method':<20} {'Cost':<10} {'Feasible':<10} {'Time (s)':<10}")
        # print("-" * 55)
        # for r in results:
        #     print(f"{r['Method']:<20} {r['Cost']:<10.2f} {r['Feasible']:<10} {r['Time']:<10.4f}")

        total_time = time.time() - start_time
        # print(f"\nTotal time: {total_time:.4f}s")
        # print(f"Best solution cost: {best_cost}")

        return best_solution


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    filename = 'scp63.txt'

    # Run metaheuristics
    solver = SCPMetaheuristics(filename)
    best_solution = solver.run_all_metaheuristics(time_limit=20)

    # Get details of best solution
    cost, feasible, _ = solver.evaluate_solution(best_solution)
    print(f"\nBest solution found with cost: {cost}")
    # print(f"Feasible: {feasible}")
    selected_atms = [atm_id for atm_id, val in best_solution.items() if val == 1]
    print(f"Selected ATMs: {len(selected_atms)} out of {len(solver.all_atms)}")
    print(f"Selected ATMS: {selected_atms}")