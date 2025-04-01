from models.user import User

def test_user_creation():
    user = User(40.7128, -74.0060, 10)
    assert user.latitude == 40.7128, "test_user_creation failed: Latitude mismatch"
    assert user.longitude == -74.0060, "test_user_creation failed: Longitude mismatch"
    assert user.maxDistance == 10, "test_user_creation failed: Max distance mismatch"

def test_user_repr():
    user = User(40.7128, -74.0060, 10)
    assert repr(user) == "User(latitude=40.7128, longitude=-74.006, maxDistance=10)", "test_user_repr failed: Repr mismatch"

def test_user_str():
    user = User(40.7128, -74.0060, 10)
    assert str(user) == "User is at location (40.7128, -74.006) and wants to travel up to 10 km.", "test_user_str failed: Str mismatch"

def test_user_setters():
    user = User(40.7128, -74.0060, 10)
    user.latitude = 41.0
    user.longitude = -75.0
    user.maxDistance = 20
    assert user.latitude == 41.0, "test_user_setters failed: Latitude setter failed"
    assert user.longitude == -75.0, "test_user_setters failed: Longitude setter failed"
    assert user.maxDistance == 20, "test_user_setters failed: Max distance setter failed"

if __name__ == '__main__':
    try:
        test_user_creation()
        test_user_repr()
        test_user_str()
        test_user_setters()
        print("All tests passed.")
    except AssertionError as e:
        print(f"Test failed: {e}")
