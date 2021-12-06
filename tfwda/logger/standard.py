import abc
from abc import ABCMeta, abstractmethod

from tfwda.utils.coloring import TerminalColors


class IFLogger(metaclass=ABCMeta):
    @abstractmethod
    def log(self, message: str, status: str):
        pass


class Logger(IFLogger):
    """
    The Logger logs to the console different information, warnings, errors, etc...
    Input:
        - verbosity : bool -> Verbosity of the information (On/Off)
    """


    def __init__(self, verbosity: bool) -> None:
        self.verbosity = verbosity


    def log(self, message: str, status: str) -> None:
        """ Prints to the console messages and its corresponding status """
        if status not in ["Header", "Info", "Warning", "Error"]:
            raise ValueError('status has to be of one of the following types: Header, Info, Warning, Error')
        if not self.verbosity:
            return
        if status == "Header":
            print(f"{TerminalColors.HEADER}{message.upper()}{TerminalColors.ENDC}")
        if status == "Info":
            print(f"{TerminalColors.INFO}{message}{TerminalColors.ENDC}")
        if status == "Warning":
            print(f"{TerminalColors.WARNING}{message}{TerminalColors.ENDC}")
        if status == "Error":
            print(f"{TerminalColors.ERROR}{message}{TerminalColors.ENDC}")