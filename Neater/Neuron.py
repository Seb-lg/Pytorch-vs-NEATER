import uuid
import numpy
import random


class Nucleus():
    def __init__(self):
        self.ID = uuid.uuid4()
        self.value = .0
        self.prev_ = []
        self.next_ = []
    
    def activate():
        self.value = random.random()* 2.0 - 1.0
        prev_sum = .0
        for axon in self.prev_:
            prev_sum += axon.prev_.value
        
        #Activation function
        self.value = numpy.tanh(prev_sum)
    
    def get_next_nucleus():
        next_nucleus = []
        for axon in self.next_:
            next_nucleus.append(axon.next_)
        return next_nucleus
        


class Axon():
    def __init__(self):
        self.weight = .0
        self.prev_ = None
        self.next_ = None
    
    def get_value(self):
        return self.prev_.value * self.weight if self.prev_ else .0

    def randomize_weight(self):
        self.weight = random.random()

