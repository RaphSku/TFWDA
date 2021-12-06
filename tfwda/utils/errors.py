class NotCreatedInstanceError(Exception):
    """
    This Error should be raised when someone wants
    to access a Singleton Class and an instance is 
    for some reason non existent or not able to be created.
    """

    def __init__(self, message) -> None:
        super().__init__(message)