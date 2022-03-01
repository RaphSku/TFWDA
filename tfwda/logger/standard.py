import abc
from abc import abstractmethod


import utils.coloring


class IFLogger(metaclass = abc.ABCMeta):
    """Interface for the logger which logs different kind of 
    information to the console

    methods (abstract)
    ------------------
        log(message str, status str)
            Logs a message to the console, the message can have
            different stati which the implementation of this interface
            defines
    """


    @abstractmethod
    def log(self, message: str, status: str):
        """Logs a message with a status to the console

        Parameters
        ----------
            message : str
                The message which should be printed
            status  : str
                The status mainly determines in which color the message
                is printed, e.g. error in red, warning in yellow
        """
        pass


class Logger(IFLogger):
    """Logs messages to the console

    Parameters
    ----------
        verbosity : bool
            Verbosity of the information

    Methods
    -------
        log(message str, status str)
            Logs a message to the console, the message can have
            different stati, like warning, error, etc.
    """


    def __init__(self, verbosity: bool):
        self.verbosity = verbosity


    def log(self, message: str, status: str) -> None:
        """Prints to the console messages and its corresponding status
        
        Parameters
        ----------
            message : str
                The message wich should be logged
            status  : str
                The status of the message

        Raises
        ------
            ValueError
                Is triggered when the status does not match one of the defined stati
        """
        if status not in ["Header", "Info", "Warning", "Error"]:
            raise ValueError('Status has to be of one of the following types: Header, Info, Warning, Error!')
        if not self.verbosity:
            return
        if status == "Header":
            print(f"{utils.coloring.TerminalColors.HEADER}{message.upper()}{utils.coloring.TerminalColors.ENDC}")
        if status == "Info":
            print(f"{utils.coloring.TerminalColors.INFO}{message}{utils.coloring.TerminalColors.ENDC}")
        if status == "Warning":
            print(f"{utils.coloring.TerminalColors.WARNING}{message}{utils.coloring.TerminalColors.ENDC}")
        if status == "Error":
            print(f"{utils.coloring.TerminalColors.ERROR}{message}{utils.coloring.TerminalColors.ENDC}")