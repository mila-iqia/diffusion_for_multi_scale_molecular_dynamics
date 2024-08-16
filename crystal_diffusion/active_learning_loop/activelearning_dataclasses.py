from dataclasses import dataclass


@dataclass(kw_only=True)
class ActiveLearningDataArguments:
    training_data_dir: str  # training data directory
    evaluation_data_dir: str  # evaluation data directory
    output_dir: str  # directory where to save the results


@dataclass(kw_only=True)
class StructureEvaluationArguments:
    evaluation_criteria: str ='nbh_grades'
    criteria_threshold: float = 10
    number_of_structures: int = None
    extraction_radius: float = 3


@dataclass(kw_only=True)
class RepaintingArguments:
    model: str = 'dev_dummy'
