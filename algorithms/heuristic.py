from typing import List
from models.atm import ATM
from models.user import User


def heuristic1(users: List[User], atms: List[ATM]) -> List[bool]:
    """
    Algorithm that chooses ATMs from the list until all users are covered.
    The ATMs are sorted in descending order of (Number of users ATM covers) / (Cost of ATM).

    :param users: List of User objects.
    :param atms: List of ATM objects.
    :return: A list of booleans indicating whether each ATM is selected.
    """
    coverage = {atm.name: sum(1 for user in users if user.is_covered_by(atm)) for atm in atms}

    atmsSorted = sorted(atms, key=lambda atm: coverage[atm.name] / atm.cost, reverse=True)

    selectedAtms = []
    uncoveredUser = set(users)

    for atm in atmsSorted:
        if not uncoveredUser:
            break
        covered_now = {user for user in uncoveredUser if user.is_covered_by(atm)}
        if covered_now:
            selectedAtms.append(atm)
            uncoveredUser -= covered_now

    return selectedAtms
