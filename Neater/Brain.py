import logging
from .BrainZone import BrainZone
from .Neuron import Nucleus, Axon


class Brain():
    def __init__(self, input_size, output_size, zone_number):
        self.zones = []
        self.neurons = []
        self.input_neurons = []
        self.output_neurons = []

        logging.debug("Constructor Brain")
        
        #Init input layer
        for i in range(input_size):
            tmp = Nucleus()
            self.input_neurons.append(tmp)
            self.neurons.append(tmp)
        
        #Init output layer
        for i in range(output_size):
            tmp = Nucleus()
            self.output_neurons.append(tmp)
            self.neurons.append(tmp)

        #Init BrainZones
        for i in range(zone_number):
            self.zones.append(BrainZone())