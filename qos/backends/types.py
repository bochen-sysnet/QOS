from abc import ABC, abstractmethod


class QPU(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @abstractmethod
    def transpile(self, *args, **kwargs):
        pass
