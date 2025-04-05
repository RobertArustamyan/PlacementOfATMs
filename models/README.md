# Models

This folder contains classes which represent **ATMs** and **Users** within the context of an ATM placement optimization
system.

## File Structure

### 1. `atm.py`

Defines the **ATM** class that represents an ATM's characteristics, such as name, geographic coordinates, cost, and
capacity limit. The ATM class also includes methods for retrieving and updating these attributes, as well as string
representations of the ATM object.

### 2. `user.py`

Defines the **User** class that represents a user's location and their maximum travel distance to an ATM. The class
provides properties for latitude, longitude, and maximum distance, as well as a method to check if a user is within the
service area of a given ATM.
