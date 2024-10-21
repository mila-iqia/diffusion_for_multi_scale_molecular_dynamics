from dataclasses import dataclass


@dataclass(kw_only=True)
class ActiveLearningDataArguments:
    """Paths to the training, validaition datasets and output directory."""
    training_data_dir: str  # training data directory
    evaluation_data_dir: str  # evaluation data directory
    output_dir: str  # directory where to save the results


@dataclass(kw_only=True)
class StructureEvaluationArguments:
    """Parameters related to the MLIP evaluation."""
    evaluation_criteria: str = 'nbh_grades'
    criteria_threshold: float = 10
    number_of_structures: int = None
    extraction_radius: float = 3


@dataclass(kw_only=True)
class RepaintingArguments:
    """Parameters related to the structure generation model."""
    model: str = 'dev_dummy'
