import math
from gurobipy import *
from utils.scp_data_reading import read_scp_raw_data, convert_scp_data_to_objects
import os
from dotenv import load_dotenv

class Result:
    def __init__(self, oV=None, variables=None):
        self._oV = oV if oV is not None else float('+inf')
        self._variables = variables if variables is not None else []

    def getOV(self):
        return self._oV

    def getVar(self):
        return self._variables

class BranchAndBoundSCP:
    epsilon = 1e-4

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
        self.model = Model("BranchAndBoundDepth", env=self.env)

        self.set_costs, self.user_coverage = read_scp_raw_data(filename)

        self.atm_list = convert_scp_data_to_objects(self.set_costs, self.user_coverage)

        # Variables: x_j = 1 if ATM j is used, 0 otherwise
        self.vars = []
        for atm in self.atm_list:
            var = self.model.addVar(vtype=GRB.BINARY, name=f"x_{atm.id}")
            self.vars.append(var)
        # Objective: minimize total cost
        self.model.setObjective(
            quicksum(atm.cost * self.vars[j] for j, atm in enumerate(self.atm_list)),
            GRB.MINIMIZE
        )

        # Constraints: each user must be covered by at least one selected ATM
        for user_id, covering_sets in self.user_coverage.items():
            self.model.addConstr(
                quicksum(self.vars[atm_idx - 1] for atm_idx in covering_sets) >= 1,
                name=f"user_{user_id}_covered"
            )

        self.model.update()
        self.model.write(lp_file_path)

    def branchAndBound(self):
        self.model.optimize()

        if self.model.Status == GRB.OPTIMAL:
            allInteger = True
            for var in self.vars:
                if abs(var.X - round(var.X)) > self.epsilon:
                    allInteger = False
                    break

            if allInteger:
                solution = [round(var.X) for var in self.vars]
                objVal = self.model.ObjVal
                return Result(objVal, solution)
            else:
                # Branch on first fractional variable
                for i, var in enumerate(self.vars):
                    if abs(var.X - round(var.X)) > self.epsilon:
                        # Left branch: var <= floor
                        modelLeft = BranchAndBoundSCP(self.user_coverage, self.atm_list, self.modelNo + 1)
                        modelLeft.model.addConstr(self.vars[i] <= 0, name=f"branch_left_{i}")

                        # Right branch: var >= ceil
                        modelRight = BranchAndBoundSCP(self.user_coverage, self.atm_list, self.modelNo + 1)
                        modelRight.model.addConstr(self.vars[i] >= 1, name=f"branch_right_{i}")

                        resultLeft = modelLeft.branchAndBound()
                        resultRight = modelRight.branchAndBound()

                        if resultLeft is None and resultRight is None:
                            return None
                        if resultLeft is None:
                            return resultRight
                        if resultRight is None:
                            return resultLeft
                        return resultLeft if resultLeft.getOV() < resultRight.getOV() else resultRight
        else:
            return None


if __name__ == '__main__':
    result = BranchAndBoundSCP("scp41.txt").branchAndBound()
    if result:
        print(f"Optimal cost: {result.getOV()}")
        print(f"ATM usage: {result.getVar()}")
        print(f"Total number of used ATM's: {sum(result.getVar())}")
    else:
        print("No feasible solution found.")
