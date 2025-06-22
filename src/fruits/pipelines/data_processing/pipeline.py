# src/fruits/pipelines/data_processing/pipeline.py

from kedro.pipeline import Pipeline, node
from .nodes import download_and_prepare_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=download_and_prepare_data,
                inputs="params:data_processing",
                outputs=["train_df", "test_df", "label_map"],
                name="download_and_prepare_data",
            ),
        ]
    )