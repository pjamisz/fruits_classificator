# src/fruits/pipelines/modeling/pipeline.py

from kedro.pipeline import Pipeline, node
from .nodes import train_fruit_classifier, evaluate_model, save_model_artifacts


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_fruit_classifier,
                inputs=["train_df", "params:modeling"],
                outputs="fruit_classifier_model",
                name="train_fruit_classifier",
            ),
            node(
                func=evaluate_model,
                inputs=["fruit_classifier_model", "test_df", "label_map"],
                outputs="evaluation_results",
                name="evaluate_model",
            ),
            node(
                func=save_model_artifacts,
                inputs=["fruit_classifier_model", "evaluation_results", "params:modeling"],
                outputs="model_path",
                name="save_model_artifacts",
            ),
        ]
    )