import math
from dotenv import load_dotenv
import os
from utils.users_living_quadrilateral_reading import get_living_quadrilateral_data
from typing import Tuple, List


def latlon_to_xy(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """
    Converts latitude and longitude coordinates to Cartesian coordinates (x, y) in meters.

    :param lat: Latitude of the point to convert (in decimal degrees).
    :param lon: Longitude of the point to convert (in decimal degrees).
    :param lat0: Latitude of the origin point (in decimal degrees).
    :param lon0: Longitude of the origin point (in decimal degrees).
    :return: A tuple (x, y) representing the Cartesian coordinates in meters.

    This function uses the Haversine formula to convert geographic coordinates (latitude and longitude)
    to Cartesian coordinates (x, y) relative to an origin point (lat0, lon0). The radius of the Earth is assumed to be 6,371 km.
    """
    R = 6371000  # Earth's radius in meters
    lat = math.radians(lat)
    lon = math.radians(lon)
    lat0 = math.radians(lat0)
    lon0 = math.radians(lon0)

    # Calculate Cartesian coordinates
    x = R * (lon - lon0) * math.cos((lat + lat0) / 2)
    y = R * (lat - lat0)
    return x, y


def area_of_quadrilateral(latlons: List[Tuple[float, float]]) -> float:
    """
    Calculates the area of a quadrilateral on the Earth's surface based on the latitudes and longitudes of its vertices.

    :param latlons: A list of 4 tuples representing the latitude and longitude of each vertex of the quadrilateral.
    :return: The area of the quadrilateral in square kilometers.

    This function uses the shoelace formula to compute the area of a quadrilateral on a spherical surface.
    The coordinates are first converted to Cartesian coordinates, and the area is calculated in square meters,
    then converted to square kilometers.
    """
    lat0, lon0 = latlons[0]
    # Converting coordinates to Cartesian coordinates
    xy = [latlon_to_xy(lat, lon, lat0, lon0) for lat, lon in latlons]
    xy.append(xy[0])

    area = 0.5 * abs(
        sum(xy[i][0] * xy[i + 1][1] - xy[i + 1][0] * xy[i][1] for i in range(4))
    )
    return area / 1_000_000  # Convert square meters to square kilometers


if __name__ == "__main__":
    load_dotenv()

    test_data_name = 'LeobenUserLivingSquares2Test.csv'
    data = get_living_quadrilateral_data(fr"{os.getenv('USERS_DATA_PATH')}\{test_data_name}")

    results = {key: area_of_quadrilateral(value) for key, value in data.items()}
    print(results)
