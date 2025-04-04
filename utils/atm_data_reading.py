from models.atm import ATM
from typing import List
import os
import csv


def get_atm_data(path: str) -> List[ATM]:
    """
    Fetches and parses ATM data from the specified file path.

    This function reads ATM data from a file at the given path and returns a list
    of ATM objects. The function is expected to process the file contents,
    parse the data correctly, and construct ATM objects accordingly.

    :param path: The file path of the data source containing ATM information.
    :type path: str
    :return: A list of ATM objects derived from the provided file data.
    """
    ATMs = []

    if not os.path.exists(path):
        print(f"File at {path} does not exist.")
        return ATMs

    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for ind,row in enumerate(reader):
            ATMs.append(ATM(
                name=row[1],
                latitude=float(row[0].split('(')[1].split(' ')[1][:-1]),
                longitude=float(row[0].split('(')[1].split(' ')[0]),
                cost=None,
                capacityLimit=None
            ))
    return ATMs

if __name__ == "__main__":
    print(get_atm_data("C:/Users/User/PycharmProjects/PlacementOfATMs/ATMsData/Leoben-73ATMsTEST.csv"))