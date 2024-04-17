from typing import Any, AnyStr, Dict

from pytorch_lightning import Callback

from crystal_diffusion.callbacks.sampling_callback import \
    instantiate_diffusion_sampling_callback
from crystal_diffusion.callbacks.standard_callbacks import (
    CustomProgressBar, instantiate_early_stopping_callback,
    instantiate_model_checkpoint_callbacks)

OPTIONAL_CALLBACK_DICTIONARY = dict(early_stopping=instantiate_early_stopping_callback,
                                    model_checkpoint=instantiate_model_checkpoint_callbacks,
                                    diffusion_sampling=instantiate_diffusion_sampling_callback)


def create_all_callbacks(hyper_params: Dict[AnyStr, Any], output_directory: str, verbose: bool) -> Dict[str, Callback]:
    """Create all callbacks.

    This method leverages the global dictionary OPTIONAL_CALLBACK_DICTIONARY which should be used to
    register all available optional callbacks and provide a standardized interface to intialize these callbacks.

    The instantiation methods can define sane defaults or hardcode constant choices.

    Args:
        hyper_params : configuration parameters.
        output_directory: path to where outputs are to be written.
        verbose: if relevant, should the callback produce verbose output.

    Returns:
        all_callbacks_dict: a dictionary of instantiated callbacks with relevant names as keys.
    """
    # We always need a progress bar.
    all_callbacks_dict = dict(progress_bar=CustomProgressBar())

    for callback_name, instantiate_callback in OPTIONAL_CALLBACK_DICTIONARY.items():
        if callback_name not in hyper_params:
            continue
        callback_params = hyper_params[callback_name]
        callback_dict = instantiate_callback(callback_params, output_directory, verbose)
        all_callbacks_dict.update(callback_dict)

    return all_callbacks_dict
