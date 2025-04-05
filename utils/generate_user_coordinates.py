import math
from models.user import User
from utils.users_living_quadrilateral_reading import get_living_quadrilateral_data
from typing import List, Tuple
import random
from utils.ordering import order_points
import os
from dotenv import load_dotenv


def generate_users_in_triangle(coordinates: List[Tuple[float, float]]) -> User:
    """
    Generates a random user uniformly located within a triangle defined by 3 geographic coordinates.

    :param coordinates: A list of 3 (latitude, longitude) tuples representing the vertices of the triangle.
    :return: A User object with a randomly generated latitude and longitude within the triangle,
             and maxDistance set to 0.

    The function uses two random numbers to create barycentric coordinates, adjusts them to ensure
    uniform distribution inside the triangle, computes the point's latitude and longitude,
    and rounds the coordinates to 7 decimal places.
    """
    random_number_1 = random.random()
    random_number_2 = random.random()

    if random_number_1 + random_number_2 > 1:
        random_number_1, random_number_2 = 1 - random_number_1, 1 - random_number_2

    user_latitude = (1 - random_number_1 - random_number_2) * coordinates[0][0] + random_number_1 * coordinates[1][
        0] + random_number_2 * coordinates[2][0]
    user_longitude = (1 - random_number_1 - random_number_2) * coordinates[0][1] + random_number_1 * coordinates[1][
        1] + random_number_2 * coordinates[2][1]

    return User(latitude=round(user_latitude, 7), longitude=round(user_longitude, 7), maxDistance=0)


def generate_users_in_quadrilateral(quadrilateral_coordinates: List[Tuple[float, float]], number: int):
    """
    Generates a specified number of random users uniformly located within a quadrilateral by dividing
    the quadrilateral into two triangles and sampling from each triangle.

    :param quadrilateral_coordinates: A list of 4 (latitude, longitude) tuples representing the vertices
                                      of the quadrilateral.
    :param number: The total number of users to generate.
    :return: A list of User objects with randomly generated coordinates within the quadrilateral,
             each having maxDistance set to 0.

    The function divides the quadrilateral into two triangles by selecting points and
    generating users for each triangle. It ensures a uniform distribution of users by sampling
    from both triangles and appending the generated users to a list.
    """
    users = []

    for i in range(number // 2):
        ordered_points = order_points(quadrilateral_coordinates)
        users.append(generate_users_in_triangle([ordered_points[0], ordered_points[1], ordered_points[2]]))
        users.append(generate_users_in_triangle([ordered_points[0], ordered_points[2], ordered_points[3]]))
    return users


if __name__ == '__main__':
    load_dotenv()

    test_data_name = 'LeobenUserLivingSquaresTest.csv'
    data = get_living_quadrilateral_data(fr"{os.getenv('USERS_DATA_PATH')}\{test_data_name}")

    # print(generate_users_in_quadrilateral(data[1],10))
    # print(order_points(data[1]))
    print(generate_users_in_quadrilateral(data[1], 1000))
