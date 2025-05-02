import os
from dotenv import load_dotenv

load_dotenv()

def read_scp_data(filename):
    folder_path = os.getenv("SCP_DATA_PATH")

    with open(os.path.join(folder_path, filename), 'r') as file:
        lines = file.readlines()
        num_users, num_sets = map(int, lines[0].split())

        set_costs = []
        i = 1
        while len(set_costs) < num_sets:
            set_costs.extend(map(int, lines[i].split()))
            i += 1

        user_coverage = {}
        for user_id in range(1, num_users + 1):
            num_covering_sets = int(lines[i])
            i += 1
            covering_sets = []
            while len(covering_sets) < num_covering_sets:
                covering_sets.extend(map(int, lines[i].split()))
                i += 1
            user_coverage[str(user_id)] = covering_sets

        return user_coverage, set_costs

if __name__ == "__main__":
    users, costs = read_scp_data("scp41.txt")
    print(f"User 1 is covered by sets: {users['1']}")
