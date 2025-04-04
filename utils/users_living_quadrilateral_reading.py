from typing import Dict, List, Tuple, Optional
import os
import csv
import re

def get_point_data(point_str: str) -> Optional[Tuple[float, float]]:
    """
    Extracts coordinate data from a 'POINT (x y)' formatted string.

    This function parses a coordinate string in the format 'POINT (x y)'
    and returns a tuple of float values representing the latitude and longitude.

    :param point_str: A string containing a point in the form   at 'POINT (x y)'.
    :type point_str: str
    :return: A tuple (x, y) with floating point values, or None if parsing fails.
    :rtype: Optional[Tuple[float, float]]
    """
    match = re.match(r"POINT \(([-\d\.]+) ([-\d\.]+)\)", point_str)
    if match:
        return float(match.group(2)), float(match.group(1))
    return None

def get_living_quadrilateral_data(path: str) -> Dict[int, List[Tuple[float, float]]]:
    """
    Reads and organizes quadrilateral coordinate data from a CSV file.

    This function reads a CSV file containing coordinate points and their
    corresponding group identifiers. It processes the data and groups
    the coordinates into quadrilaterals based on the identifier's prefix.

    :param path: The file path of the CSV file containing coordinate data.
    :type path: str
    :return: A dictionary where keys represent quadrilateral group numbers
             and values are lists of (x, y) coordinate tuples.
    :rtype: Dict[int, List[Tuple[float, float]]]
    """
    quadrilateral_coordinates: Dict[int, List[Tuple[float, float]]] = {}

    if not os.path.exists(path):
        print(f"File at {path} does not exist.")
        return quadrilateral_coordinates

    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)

        for row in reader:
            point, identifier = row[:2]
            coordinates = get_point_data(point)

            if coordinates:
                group_id = int(identifier.split('-')[0])
                if group_id not in quadrilateral_coordinates:
                    quadrilateral_coordinates[group_id] = []
                quadrilateral_coordinates[group_id].append(coordinates)

    return quadrilateral_coordinates

if __name__ == "__main__":
    data = get_living_quadrilateral_data("C:/Users/User/PycharmProjects/PlacementOfATMs/ATMsData/LeobenUserLivingSquares.csv")
    print(data)
