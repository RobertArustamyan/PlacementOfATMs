class User:
    def __init__(self, latitude, longitude, maxDistance):
        self.__latitude = latitude
        self.__longitude = longitude
        self.__maxDistance = maxDistance

    def getLatitude(self):
        return self.__latitude

    def getLongitude(self):
        return self.__longitude

    def getMaxDistance(self):
        return self.__maxDistance

    def setLatitude(self, latitude):
        self.__latitude = latitude

    def setLongitude(self, longitude):
        self.__longitude = longitude

    def setMaxDistance(self, maxDistance):
        self.__maxDistance = maxDistance
