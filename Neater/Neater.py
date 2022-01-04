import logging
from .Genus import Genus


class Neater():
    def __init__(self, input_size, output_size, max_genus=20):
        self.input_size = input_size
        self.output_size = output_size
        self.max_genus = max_genus

        logging.info("Constructor Neater")
        self.genus = []
        for i in range(max_genus):
            self.genus.append(Genus(self.input_size, self.output_size))


def train_neater(train, test):
    test = Neater()