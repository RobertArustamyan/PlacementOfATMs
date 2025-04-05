from math import radians, sin, cos, sqrt, atan2


def haversine(latitude1, longitude1, latitude2, longitude2):
    EARTH_RADIUS_KM = 6371  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [latitude1, longitude1, latitude2, longitude2])

    # Convert latitude and longitude from degrees to radians
    lat1_rad, lon1_rad = radians(latitude1), radians(longitude1)
    lat2_rad, lon2_rad = radians(latitude2), radians(longitude2)

    # Compute differences
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    # Haversine formula
    a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Compute distance
    distance_km = EARTH_RADIUS_KM * c
    return distance_km  # Returns the distance in kilometers


# Testing haversine
if __name__ == '__main__':
    leoben_latitude = 47.3827417
    leoben_longitude = 15.0942137

    vienna_latitude = 48.210033
    vienna_longitude = 16.363449

    print(haversine(leoben_latitude, leoben_longitude, vienna_latitude, vienna_longitude))
