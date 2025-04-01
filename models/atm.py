class ATM:
    def __init__(self, latitude, longitude, cost, capacityLimit):
        self.__latitude = latitude
        self.__longitude = longitude
        self.__cost = cost
        self.__capacityLimit = capacityLimit

    def __repr__(self):
        return f"ATM(latitude={self.__latitude}, longitude={self.__longitude}, cost={self.__cost}, capacityLimit={self.__capacityLimit})"

    def __str__(self):
        return f"ATM is located at ({self.__latitude}, {self.__longitude}) and has a cost of {self.__cost} and a capacity limit of {self.__capacityLimit}."

    @property
    def latitude(self):
        return self.__latitude

    @property
    def longitude(self):
        return self.__longitude

    @property
    def cost(self):
        return self.__cost

    @property
    def capacityLimit(self):
        return self.__capacityLimit

    @latitude.setter
    def latitude(self, latitude):
        self.__latitude = latitude

    @longitude.setter
    def longitude(self, longitude):
        self.__longitude = longitude

    @cost.setter
    def cost(self, cost):
        self.__cost = cost

    @capacityLimit.setter
    def capacityLimit(self, capacityLimit):
        self.__capacityLimit = capacityLimit
