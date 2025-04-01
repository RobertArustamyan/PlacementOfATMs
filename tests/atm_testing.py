# test/test_atm.py

from models.atm import ATM

def test_atm_creation():
    # Test the creation of an ATM object
    atm = ATM(40.7128, -74.0060, 1000, 500)
    assert atm.latitude == 40.7128, "Latitude mismatch"
    assert atm.longitude == -74.0060, "Longitude mismatch"
    assert atm.cost == 1000, "Cost mismatch"
    assert atm.capacityLimit == 500, "Capacity limit mismatch"

def test_atm_repr():
    # Test the __repr__ method
    atm = ATM(40.7128, -74.0060, 1000, 500)
    assert repr(atm) == "ATM(latitude=40.7128, longitude=-74.006, cost=1000, capacityLimit=500)", "Repr mismatch"

def test_atm_str():
    # Test the __str__ method
    atm = ATM(40.7128, -74.0060, 1000, 500)
    assert str(atm) == "ATM is located at (40.7128, -74.006) and has a cost of 1000 and a capacity limit of 500.", "Str mismatch"

def test_atm_setters():
    # Test setters
    atm = ATM(40.7128, -74.0060, 1000, 500)
    atm.latitude = 41.0
    atm.longitude = -75.0
    atm.cost = 1200
    atm.capacityLimit = 600
    assert atm.latitude == 41.0, "Latitude setter failed"
    assert atm.longitude == -75.0, "Longitude setter failed"
    assert atm.cost == 1200, "Cost setter failed"
    assert atm.capacityLimit == 600, "Capacity limit setter failed"

if __name__ == '__main__':
    try:
        test_atm_creation()
        test_atm_repr()
        test_atm_str()
        test_atm_setters()
        print("All tests passed.")
    except AssertionError as e:
        print(f"Test failed: {e}")
