import os
from operator import truediv
import pandas as pd
import numpy as np
import time
from gurobipy import Env, Model, GRB, LinExpr
from utils.scp_data_reading import read_scp_raw_data, convert_scp_data_to_objects


class Result:
    def __init__(self, oV=None, variables=None):
        self._oV = oV if oV is not None else float('+inf')
        self._variables = variables if variables is not None else {}

    def getOV(self):
        return self._oV

    def getVar(self):
        return self._variables

class BranchAndBoundSCP:
    # tolerance for pruning
    GAP_TOLERANCE = 1e-10
    upper_bound = float('+inf')
    lower_bound = 0.0
    terminate = False
    best_solution = None

    def __init__(self, filename, modelNo=0):
        self.modelNo = modelNo
        self.filename = filename

        basename = os.path.splitext(os.path.basename(filename))[0]
        log_file_path = os.getenv("LOG_FOLDER_PATH") + fr"\BranchAndBoundDepth_{basename}.log"
        lp_file_path = os.getenv("LP_MODELS_FOLDER_PATH") + fr"\SCP_LP_{basename}_{modelNo}.lp"

        self.env = Env(empty=True)
        self.env.setParam("LogFile", log_file_path)
        self.env.setParam("OutputFlag", 0)
        self.env.start()

        self.set_costs, self.user_coverage = read_scp_raw_data(filename)
        self.atm_list = convert_scp_data_to_objects(self.set_costs, self.user_coverage)

        self.best_solution = {atm.id: 0 for atm in self.atm_list}
        self.best_solution_value = float("+inf")
        # self.model = Model(env=self.env)
        self.time_to_solve = None

    def solve(self,use_heuristic=True):
        if use_heuristic:
            self.greedy_heuristic()

        start = time.time()
        fixed0 = {atm.id: False for atm in self.atm_list}
        fixed1 = {atm.id: False for atm in self.atm_list}
        end = time.time()

        self.time_to_solve = end - start
        self._branch_and_bound(fixed0, fixed1)
        return Result(self.best_solution_value, self.best_solution)

    def _branch_and_bound(self, fixedTo0, fixedTo1):
        m = Model(env=self.env)
        x = {}
        # print('relaxation')
        for atm in self.atm_list:
            lb, ub = 0.0, 1.0
            if fixedTo0[atm.id]: ub = 0.0
            if fixedTo1[atm.id]: lb = 1.0
            x[atm.id] = m.addVar(lb=lb, ub=ub, obj=atm.cost, vtype=GRB.CONTINUOUS, name=f'x_{atm.id}')

        for cover in self.user_coverage.values():
            expr = LinExpr()
            for j in cover:
                expr.addTerms(1.0, x[j])
            m.addConstr(expr >= 1.0)

        m.modelSense = GRB.MINIMIZE
        m.update()
        m.optimize()


        if m.status != GRB.OPTIMAL:
            # print("No optimal LP relaxation solution found.")
            m.dispose()
            return

        if  m.objVal > self.best_solution_value:
            # print("Skipping: Relaxation is worse than the current best solution.")
            m.dispose()
            return

        frac = [(index, value.x) for index, value in x.items()
                if not np.isclose(value.x, 0.0, atol=self.GAP_TOLERANCE) and not np.isclose(value.x, 1.0,
                                                                                            atol=self.GAP_TOLERANCE)]
        if not frac:
            # print('found fisible')
            # Found a feasible integer solution
            sol = {i: int(round(v.x)) for i, v in x.items()}
            obj_val = sum(sol[atm.id] * atm.cost for atm in self.atm_list)

            if obj_val < self.best_solution_value:
                self.best_solution_value = obj_val
                self.best_solution = sol.copy()
                # print(f"New best solution found with cost: {self.best_solution_value}")
            # else:
                # print(f"Found feasible but solution is worse: {obj_val}")
                # print(f"Best solution so far: {self.best_solution_value}")
            m.dispose()
            return

        branch_id, branch_value = min(frac, key=lambda item: abs(0.5 - item[1]))
        # branch_id, branch_value = frac[0]
        if branch_value >= 0.5:
            fixedTo1[branch_id] = True
            self._branch_and_bound(fixedTo0,fixedTo1)
            fixedTo1[branch_id] = False

            fixedTo0[branch_id] = True
            self._branch_and_bound(fixedTo0, fixedTo1)
            fixedTo0[branch_id] = False
        else:
            fixedTo0[branch_id] = True
            self._branch_and_bound(fixedTo0, fixedTo1)
            fixedTo0[branch_id] = False

            fixedTo1[branch_id] = True
            self._branch_and_bound(fixedTo0, fixedTo1)
            fixedTo1[branch_id] = False

    def greedy_heuristic(self):
        uncovered_users = set(self.user_coverage.keys())
        selected_atms = set()
        total_cost = 0

        while uncovered_users:
            best_atm = None
            best_score = float('-inf')

            for atm in self.atm_list:
                if atm.id in selected_atms:
                    continue  # already selected

                # how many uncovered users this ATM covers
                covers = uncovered_users.intersection(atm.covered_users_ids)
                if not covers:
                    continue

                # score: number of new users covered per unit cost
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

        # Store as the initial best solution for BnB
        sol = {atm.id: 1 if atm.id in selected_atms else 0 for atm in self.atm_list}
        self.best_solution_value = total_cost
        self.best_solution = sol


def find_users_covered(selected_atms,all_atm_list):
    selected_atm_objects = [atm for atm in all_atm_list if atm.id in selected_atms]
    cost = 0
    all_users = []
    for atm in selected_atm_objects:
        all_users.extend(atm.covered_users_ids)
        cost += atm.cost
    print(f"Total users covered: {len(set(all_users))}")

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    # filenames = ['scp41.txt', 'scp42.txt', 'scp43.txt', 'scp44.txt', 'scp45.txt',
    #              'scp46.txt', 'scp47.txt', 'scp48.txt', 'scp49.txt', 'scp51.txt',
    #              'scp52.txt', 'scp53.txt', 'scp57.txt', 'scp61.txt', 'scp62.txt']
    # tactic_name = "DFSWithPriority"
    #
    # excel_file_path = 'BranchAndBound_tactic_results.xlsx'
    # results = {filename: [] for filename in filenames}
    #
    # for filename in filenames:
    #     for i in range(500):
    #         start = time.time()
    #
    #         solver = BranchAndBoundSCP(filename)
    #         result = solver.solve()
    #
    #         elapsed_time = time.time() - start
    #         results[filename].append(elapsed_time)
    #
    # row = [tactic_name]
    # for filename in filenames:
    #     times = results[filename]
    #     mean_time = np.mean(times)
    #     min_time = np.min(times)
    #     max_time = np.max(times)
    #     std_time = np.std(times)
    #     row.extend([mean_time, min_time, max_time, std_time])
    #
    # stats = ['Mean', 'Min', 'Max', 'Std']
    # columns = ['Tactic'] + [f'{filename}_{stat}' for filename in filenames for stat in stats]
    #
    # if os.path.exists(excel_file_path):
    #     df_existing = pd.read_excel(excel_file_path)
    # else:
    #     df_existing = pd.DataFrame(columns=columns)
    #
    # # Check if the tactic is already in the table — update or add row
    # if tactic_name in df_existing['Tactic'].values:
    #     df_existing.loc[df_existing['Tactic'] == tactic_name, :] = row
    # else:
    #     new_row = pd.DataFrame([row], columns=columns)
    #     df_existing = pd.concat([df_existing, new_row], ignore_index=True)
    #
    # # Save the updated DataFrame back to the Excel file
    # df_existing.to_excel(excel_file_path, index=False)
    #
    # print(f"Tactic '{tactic_name}' results saved to {excel_file_path}")

    solver = BranchAndBoundSCP('scp43.txt')
    results_without_heuristic = []
    results_with_heuristic = []



    result = solver.solve(use_heuristic=True)
    print(f"Time to solve with heuristic: {solver.time_to_solve:.10f}  Result: {result.getOV()}")
    result = solver.solve(use_heuristic=False)
    print(f"Time to solve without heuristic: {solver.time_to_solve:.10f}  Result: {result.getOV()}")

    # print(f"Optimal cost: {result.getOV()}")
    # print("Selected ATMs:", [aid for aid, val in result.getVar().items() if val])
    # print(f"total num of atms: {len([aid for aid, val in result.getVar().items() if val])}")

