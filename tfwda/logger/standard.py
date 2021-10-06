import abc
from abc import ABCMeta, abstractmethod
from tfwda.utils.coloring import TerminalColors


class IF_Logger(metaclass=ABCMeta):
    @abstractmethod
    def log(self, message: str):
        pass


class Logger(IF_Logger):
    def __init__(self):
        pass


    def log(self, message: str, status: str):
        if status not in ["Header", "Info", "Warning", "Error"]:
            raise ValueError('status has to be of one of the following types: Header, Info, Warning, Error')
        if status == "Header":
            print(f"{TerminalColors.HEADER}{message}{TerminalColors.ENDC}")
        if status == "Info":
            print(f"{TerminalColors.INFO}{message}{TerminalColors.ENDC}")
        if status == "Warning":
            print(f"{TerminalColors.WARNING}{message}{TerminalColors.ENDC}")
        if status == "Error":
            print(f"{TerminalColors.ERROR}{message}{TerminalColors.ENDC}")
        