# src/fruits/pipeline_registry.py

from typing import Dict
from kedro.pipeline import Pipeline

from fruits.pipelines import data_processing, modeling


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = data_processing.create_pipeline()
    modeling_pipeline = modeling.create_pipeline()
    
    return {
        "__default__": data_processing_pipeline + modeling_pipeline,
        "data_processing": data_processing_pipeline,
        "modeling": modeling_pipeline,
    }