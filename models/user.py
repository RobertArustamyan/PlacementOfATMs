from models.atm import ATM
from utils.distance import haversine


class User:
    def __init__(self, latitude: float, longitude: float, maxDistance: float) -> None:
        """
        Initialize a User object.

        :param latitude: The latitude of the user.
        :param longitude: The longitude of the user.
        :param maxDistance: The maximum distance the user is willing to travel to an ATM.
        """
        self.__latitude: float = latitude
        self.__longitude: float = longitude
        self.__maxDistance: float = maxDistance

    def __repr__(self) -> str:
        return f"User(latitude={self.__latitude}, longitude={self.__longitude}, maxDistance={self.__maxDistance})"

    def __str__(self) -> str:
        return f"User is at location ({self.__latitude}, {self.__longitude}) and wants to travel up to {self.__maxDistance} km."

    @property
    def latitude(self) -> float:
        """
        Get the latitude of the user.

        :return: The latitude as a float.
        """
        return self.__latitude

    @property
    def longitude(self) -> float:
        """
        Get the longitude of the user.

        :return: The longitude as a float.
        """
        return self.__longitude

    @property
    def maxDistance(self) -> float:
        """
        Get the maximum travel distance of the user.

        :return: The max distance in kilometers.
        """
        return self.__maxDistance
    @latitude.setter
    def latitude(self, latitude: float) -> None:
        """
        Set the latitude of the user.

        :param latitude: The new latitude value.
        """
        self.__latitude = latitude
    @longitude.setter
    def longitude(self, longitude: float) -> None:
        """
        Set the longitude of the user.

        :param longitude: The new longitude value.
        """
        self.__longitude = longitude
    @maxDistance.setter
    def maxDistance(self, maxDistance: float) -> None:
        """
        Set the maximum travel distance for the user.

        :param maxDistance: The new max distance value.
        """
        self.__maxDistance = maxDistance

    def is_covered_by(self, atm: ATM) -> bool:
        """
        Check if this user is within the max distance of the given ATM.

        :param atm: The ATM object to check coverage against.
        :return: True if the ATM is within the user's max distance, False otherwise.
        """
        return haversine(
            self.__latitude, self.__longitude,
            atm.latitude, atm.longitude
        ) <= self.__maxDistance
