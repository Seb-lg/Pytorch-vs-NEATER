import logging
from .Specie import Specie
from .Brain import Brain


class Neater():
    def __init__(self, input_size, output_size, zone_number, max_species=20, max_population=5000):
        self.input_size = input_size
        self.output_size = output_size
        self.zone_number = zone_number
        self.max_species = max_species
        self.max_population = max_population

        logging.debug("Constructor Neater")
        self.species = []
        self.species.append(Specie(self.input_size, self.output_size, self.zone_number, self.max_population))

        base_network = Brain(self.input_size, self.output_size, self.zone_number)
        self.species[0].init_population([base_network])


def train_neater(train, test):
    test = Neater(32*32, 28, 2)