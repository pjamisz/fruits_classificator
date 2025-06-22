
from autogluon.multimodal import MultiModalPredictor
import pandas as pd
from typing import Dict, Any
import uuid
import json
import os
from pathlib import Path

def train_fruit_classifier(
    train_df: pd.DataFrame,
    params: Dict
) -> MultiModalPredictor:
    """
    Train a fruit classifier using AutoGluon MultiModalPredictor.
    
    Args:
        train_df: Training DataFrame with 'image' and 'label' columns
        params: Model training parameters
        
    Returns:
        Trained MultiModalPredictor
    """
    # Set model path
    #model_path = params.get("model_path", f"data/06_models/automm_fruit_classifier_{uuid.uuid4().hex}")
    
    # Get base model path from params
    base_model_path = params.get("model_path", "data/06_models/automm_fruit_classifier")
    
    # Check if model already exists at the specified path
    if os.path.exists(base_model_path) and params.get("use_existing_model", False):
        # Load existing model
        print(f"Loading existing model from: {base_model_path}")
        predictor = MultiModalPredictor.load(base_model_path)
        return predictor
        
    # Create unique model path with timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{base_model_path}_{timestamp}"
    
    print(f"Training new model at: {model_path}")    
    # Initialize predictor
    predictor = MultiModalPredictor(
        label="label",
        path=model_path,
        problem_type="multiclass"
    )
    
    # Train the model
    predictor.fit(
        train_data=train_df,
        time_limit=params.get("time_limit", 120),  # 2 minutes default
        presets=params.get("presets", "medium_quality")
    )
    
    return predictor


def evaluate_model(
    predictor: MultiModalPredictor,
    test_df: pd.DataFrame,
    label_map: Dict
) -> Dict[str, Any]:
    """
    Evaluate the trained model on test data.
    
    Args:
        predictor: Trained MultiModalPredictor
        test_df: Test DataFrame
        label_map: Mapping of fruit names to label indices
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Evaluate on test set
    scores = predictor.evaluate(test_df, metrics=["accuracy", "balanced_accuracy"])
    
    # Get predictions and probabilities for a sample
    sample_size = min(5, len(test_df))
    sample_predictions = predictor.predict(test_df.head(sample_size))
    sample_proba = predictor.predict_proba(test_df.head(sample_size))
    
    # Reverse label map for interpretation
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    # Convert predictions to numpy array if it's a Series
    if hasattr(sample_predictions, 'values'):
        sample_predictions_array = sample_predictions.values
    else:
        sample_predictions_array = sample_predictions
    
    # Convert probabilities to numpy array if needed
    if hasattr(sample_proba, 'values'):
        sample_proba_array = sample_proba.values
    else:
        sample_proba_array = sample_proba
    
    results = {
        "test_accuracy": scores["accuracy"],
        "test_balanced_accuracy": scores.get("balanced_accuracy", scores["accuracy"]),
        "label_map": label_map,
        "sample_predictions": [
            {
                "image": test_df.iloc[i]["image"],
                "true_label": int(test_df.iloc[i]["label"]),
                "true_fruit": reverse_label_map[test_df.iloc[i]["label"]],
                "predicted_label": int(sample_predictions_array[i]),
                "predicted_fruit": reverse_label_map[int(sample_predictions_array[i])],
                "probabilities": sample_proba_array[i].tolist()
            }
            for i in range(sample_size)
        ]
    }
    
    return results
    
    
def save_model_artifacts(
    predictor: MultiModalPredictor,
    evaluation_results: Dict,
    params: Dict
) -> str:
    """
    Save model and results.
    
    Args:
        predictor: Trained model
        evaluation_results: Evaluation results
        params: Parameters
        
    Returns:
        Path to saved model
    """
    # Model is already saved during training
    # Save evaluation results
    results_path = params.get("results_path", "data/08_reporting/model_results.json")
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    return predictor.path