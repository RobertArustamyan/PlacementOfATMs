class SCPAtm:
    def __init__(self, cost_of_column: float, is_used: bool) -> None:
        self.__cost = cost_of_column
        self.__is_used = is_used
        self.__users_covered = []

    @property
    def cost(self) -> float:
        return self.__cost

    @cost.setter
    def cost(self, value: float) -> None:
        self.__cost = value

    @property
    def is_used(self) -> bool:
        return self.__is_used

    @is_used.setter
    def is_used(self, value: bool) -> None:
        self.__is_used = value

    @property
    def users_covered(self) -> list:
        return self.__users_covered

    @users_covered.setter
    def users_covered(self, users: list) -> None:
        self.__users_covered = users

    def add_user_covered(self, user_index: int) -> None:
        self.__users_covered.append(user_index)
