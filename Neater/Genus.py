import logging


class Genus():
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        
        logging.info("Constructor Genus")