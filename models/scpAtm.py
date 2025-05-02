class SCPAtm:
    def __init__(self, id, cost_of_column: float, is_used: bool = False) -> None:
        self.id = id
        self.__cost = cost_of_column
        self.__is_used = is_used
        self.__covered_users_id = []

    def __repr__(self):
        return (f"SCPAtm(id={self.id}, "
                f"cost={self.__cost}, "
                f"is_used={self.__is_used}, "
                f"covered_users_ids={self.__covered_users_id})")

    def __str__(self):
        return (f"ATM {self.id}: cost={self.__cost}, "
                f"{'used' if self.__is_used else 'not used'}, "
                f"covers {len(self.__covered_users_id)} users")

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
    def covered_users_ids(self) -> list:
        return self.__covered_users_id

    @covered_users_ids.setter
    def covered_users_ids(self, users: list) -> None:
        self.__covered_users_id = users

    def add_user_covered(self, user_index: int) -> None:
        self.__covered_users_id.append(user_index)
