from dataclasses import dataclass
from typing import Any, Dict


def create_parameters_from_configuration_dictionary(configuration: Dict[str, Any],
                                                    identifier: str,
                                                    options: Dict[str, dataclass]) -> dataclass:
    """Create Parameters from Configuration Dictionary.

    This method will instantiate a dataclass container describing configuration parameters
    from a configuration dictionary. It is assumed that this configuration dictionary contains
    an 'identifier' field to describe which parameter class should be instantiated. This identifier
    should be a key in the 'options' dictionary.

    Args:
        configuration : a dictionary containing the values needed to instantiate a parameter dataclass.
        identifier : field name of value that identifies which parameter class should be instantiated.
        options : dictionary containing all the possible classes that can be instantiated.

    Returns:
        parameters: a dataclass object of the appropriate type, instantiated with the content of the input
            configuration
    """
    assert identifier in configuration.keys(), \
        f"The identifier field '{identifier}' is missing from the configuration dictionary."

    option_id = configuration[identifier]

    assert option_id in options.keys(), \
        f"The option field '{option_id}' is missing from the options dictionary."

    parameters = options[option_id](**configuration)
    return parameters
