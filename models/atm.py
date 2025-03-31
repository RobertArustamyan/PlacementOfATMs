class ATM:
    def __init__(self, latitude, longitude, cost, capacityLimit):
        self.__latitude = latitude
        self.__longitude = longitude
        self.__cost = cost
        self.__capacityLimit = capacityLimit

    def getLatitude(self):
        return self.__latitude

    def getLongitude(self):
        return self.__longitude

    def getCost(self):
        return self.__cost

    def getCapacityLimit(self):
        return self.__capacityLimit

    def setLatitude(self, latitude):
        self.__latitude = latitude

    def setLongitude(self, longitude):
        self.__longitude = longitude

    def setCost(self, cost):
        self.__cost = cost

    def setCapacityLimit(self, capacityLimit):
        self.__capacityLimit = capacityLimit
