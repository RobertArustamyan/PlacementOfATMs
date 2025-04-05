import math
from typing import List, Tuple, Dict

def order_points(coordinates: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Orders the points of a polygon in a counter-clockwise manner based on their angle with respect to the centroid.

    :param coordinates: A list of 4 tuples, each representing the latitude and longitude of a vertex.
    :return: A list of points (latitude, longitude) ordered counter-clockwise based on their angle from the centroid.

    This function calculates the centroid of the points and then orders the points based on their angle
    from the centroid, ensuring a counter-clockwise ordering. The angle is computed using `math.atan2`,
    which calculates the arctangent of the difference in y-coordinates and x-coordinates.
    """
    Cx = sum(coordinate[0] for coordinate in coordinates) / 4
    Cy = sum(coordinate[1] for coordinate in coordinates) / 4

    points = coordinates.copy()
    points.sort(key=lambda p: math.atan2(p[1] - Cy, p[0] - Cx))
    return points


def order_point_in_dictionary(unordered_dict: Dict[int, List[Tuple[float, float]]]) -> Dict[int, List[Tuple[float, float]]]:
    """
    Orders the points in a dictionary of polygons by their counter-clockwise angle from the centroid.

    :param unordered_dict: A dictionary where keys are identifiers and values are lists of 4 points (latitude, longitude).
    :return: A dictionary with the same keys, but with the point lists ordered counter-clockwise.

    This function iterates over each item in the dictionary, applying the `order_points` function to each
    list of points to ensure they are ordered counter-clockwise based on their angle from the centroid.
    """
    for k, v in unordered_dict.items():
        unordered_dict[k] = order_points(v)
    return unordered_dict


if __name__ == '__main__':
    test_dictionary = {
        1: [(47.3880189, 15.0969993), (47.3873724, 15.0927292), (47.3818952, 15.0879251), (47.383311, 15.0952186)],
        2: [(47.3821828, 15.0904908), (47.3831562, 15.0952544), (47.380421, 15.0908781), (47.3810273, 15.0961839)]
    }

    print(order_point_in_dictionary(test_dictionary))
