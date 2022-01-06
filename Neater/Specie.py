import copy
import logging
from .Brain import Brain


class Specie():
    def __init__(self, input_number, output_number, zone_number, population_number):
        self.input_number = input_number
        self.output_number = output_number
        self.zone_number = zone_number
        self.population_number = population_number
        
        logging.debug("Constructor Specie")
        self.population = []

    def init_population(self, brains):
        logging.debug('Specie:init_population size=' + str(len(brains)))
        if len(brains) == 1:
            self.population.clear()
            logging.debug('Specie:init_population pop=' + str(self.population_number)) 
            for i in range(self.population_number):
                self.population.append(copy.deepcopy(brains[0]))
            logging.debug('Specie:init_population brains=' + str(len(self.population)))
        else:
            logging.error('Implement crossover')