from abc import ABC, abstractmethod


class VQAModule(ABC):
    @abstractmethod
    def __call__(self, images, questions):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def preprocess_input(self, images, questions):
        pass

    @abstractmethod
    def postprocess_output(self, raw_output):
        pass
