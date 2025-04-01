if __name__ == "__main__":
    try:
        import gurobipy as gp

        # Printing version of gurobi
        print(f"Gurobi version: {gp.__version__}")

        model = gp.Model("gurobi_installation_testing")
        model.setParam('OutputFlag', 0)
        # Adding test variables
        x_0 = model.addVar(name="x_0")
        x_1 = model.addVar(name="x_1")
        # Setting model's objective function
        model.setObjective(x_0 + x_1, gp.GRB.MAXIMIZE)
        # Adding constraint
        model.addConstr(x_0 + x_1 <= 4, "c_0")

        model.optimize()
        # Results
        if model.status == gp.GRB.OPTIMAL:
            print(f"Optimal objective: {model.objVal}")
            for v in model.getVars():
                print(f"{v.varName}: {v.x}")
    except gp.GurobiError as e:
        print(f"Gurobi Error: {e}")

    except Exception as e:
        print(f"Unexpected error: {e}")
