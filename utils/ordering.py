import math


def order_points(coordinates):
    Cx = sum(coordinate[0] for coordinate in coordinates) / 4
    Cy = sum(coordinate[1] for coordinate in coordinates) / 4

    points = coordinates.copy()
    points.sort(key=lambda p: math.atan2(p[1] - Cy, p[0] - Cx))
    return points


def order_point_in_dictionary(unordered_dict):
    for k,v in unordered_dict.items():
        unordered_dict[k] = order_points(v)
    return unordered_dict

if __name__ == '__main__':
    test_dictionary = {
        1: [(47.3880189, 15.0969993), (47.3873724, 15.0927292), (47.3818952, 15.0879251), (47.383311, 15.0952186)],
        2: [(47.3821828, 15.0904908), (47.3831562, 15.0952544), (47.380421, 15.0908781), (47.3810273, 15.0961839)]}

    print(order_point_in_dictionary(test_dictionary))