import os
from dotenv import load_dotenv
from models.scpAtm import SCPAtm

load_dotenv()


def read_scp_raw_data(filename):
    """Reads the SCP data file and returns raw set costs and user coverage mappings."""
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
            user_coverage[user_id] = covering_sets  # using int keys instead of str for clarity

        return set_costs, user_coverage


def convert_scp_data_to_objects(set_costs, user_coverage):
    """Creates SCPAtm objects from set costs and assigns covered users to each."""
    list_of_sets = [SCPAtm(i + 1, cost) for i,cost in enumerate(set_costs)]

    # Assign users covered by each set
    for user_id, covering_sets in user_coverage.items():
        for set_index in covering_sets:
            # Adjust index: assuming input is 1-based
            list_of_sets[set_index - 1].covered_users_ids.append(user_id)

    return list_of_sets


if __name__ == "__main__":
    set_costs, user_coverage = read_scp_raw_data("scp41.txt")
    atm_objects = convert_scp_data_to_objects(set_costs, user_coverage)

    # Example outputs
    print(atm_objects)
    print(f"User 1 is covered by sets: {user_coverage[1]}")
