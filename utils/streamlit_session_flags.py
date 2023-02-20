class Flag:

    def __init__(
        self,
        last_feeding: bool = False,
        baby_birthday: bool = False
    ):
        self.last_feeding = last_feeding
        self.baby_birthday = baby_birthday

    def set_last_feeding(self):
        self.last_feeding = True

    def set_baby_birthday(self):
        self.baby_birthday = True
