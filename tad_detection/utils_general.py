import json
import logging
import os

def load_parameters(path_parameters_json):
    '''
    Function loads the parameters from the provided parameters.json file in a dictionary.

    :param path_parameters_json: path of parameters.json file
    :return parameters: dictionary with parameters set in parameters.json file.
    '''

    with open(path_parameters_json) as parameters_json:
        parameters = json.load(parameters_json)

    return parameters

def dump_parameters(parameters):
    '''
    Function dumps the parameters from the provided parameters.json file in the dataset dictionary.

    :param parameters: dictionary with parameters set in parameters.json file.
    '''

    with open(os.path.join(parameters["output_directory"], parameters["dataset_name"], 'parameters.json'), 'w') as f:
        json.dump(parameters, f)

def set_up_logger(name, parameters):
    '''
    Function sets a global logger for documentation of information and errors in the execution of the chosen script.

    :param parameters: dictionary with parameters set in parameters.json file.
    :return :logger
    '''

    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    fh = logging.FileHandler(os.path.join(parameters["output_directory"], parameters["dataset_name"], name + '.log'))
    fh.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(fh)

    logger.info("Start " + name + " with the following parameters:")
    for parameter in parameters.keys():
        logger.info(parameter + ": " + str(parameters[parameter]))

    return logger