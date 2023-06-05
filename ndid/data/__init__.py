from abc import ABC, abstractmethod
from typing import Tuple, Callable, List

PRINT_SIZE = True
DEFAULT_BATCH_SIZE = 32


class AsbDataset(ABC):
    def __init__(self,
                 name: str,
                 classes: List[str] = None,
                 image_size: Tuple[int, int] = None,
                 num_classes: int = None,
                 batch_size: int = None,
                 map_fn: Callable = None):
        self.name = name
        self.classes = classes
        self.num_classes = num_classes
        if classes is not None:
            self.num_classes = len(classes)
        self._image_size = image_size
        self._batch_size = batch_size
        self._map_fn = map_fn

        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

    def get_classes(self):
        return self.classes

    def get_num_classes(self):
        return self.num_classes

    def get_train(self):
        if self._train_ds is None:
            self.__load()
        return self._train_ds

    def get_val(self):
        if self._val_ds is None:
            self.__load()
        return self._val_ds

    def get_test(self):
        if self._test_ds is None:
            self.__load()
        return self._test_ds

    def get_combined(self):
        return self.get_train().concatenate(self.get_val()).concatenate(self.get_test())

    def __load(self):
        train_ds, val_ds, test_ds = self._load_dataset(self._image_size, self._batch_size, self._map_fn)

        self._train_ds = train_ds
        self._val_ds = val_ds
        self._test_ds = test_ds

        if PRINT_SIZE:
            print(self.name, "dataset loaded")
            if train_ds is not None:
                print("Training size:", train_ds.cardinality().numpy())
            if val_ds is not None:
                print("Validation size:", val_ds.cardinality().numpy())
            if test_ds is not None:
                print("Evaluation size:", test_ds.cardinality().numpy())

    @abstractmethod
    def _load_dataset(self, image_size, batch_size, map_fn):
        pass
