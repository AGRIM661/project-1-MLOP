import os
import pandas as pd
import logging
import pickle

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ==========================
# Logging Configuration
# ==========================

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    os.makedirs("logs", exist_ok=True)

    file_handler = logging.FileHandler(
        "logs/model_evaluation.log"
    )
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ==========================
# Load Dataset
# ==========================

def load_data(test_path, train_path):

    try:

        df_test = pd.read_csv(test_path)
        df_train = pd.read_csv(train_path)

        logger.info(
            "Test dataset shape: %s",
            df_test.shape
        )

        logger.info(
            "Train dataset shape: %s",
            df_train.shape
        )

        logger.info(
            "Datasets loaded successfully."
        )

        return df_test, df_train

    except Exception as e:

        logger.error(
            "Failed to load data: %s",
            e
        )

        raise


# ==========================
# Load Models
# ==========================

def load_models(model_path):

    try:

        with open(
            model_path,
            "rb"
        ) as file:

            models = pickle.load(file)

        logger.info(
            "Models loaded successfully from %s",
            model_path
        )

        # Check that both models exist

        if "random_forest" not in models:

            raise KeyError(
                "Random Forest model not found in models.pkl"
            )

        if "stacking" not in models:

            raise KeyError(
                "Stacking model not found in models.pkl"
            )

        return models

    except FileNotFoundError:

        logger.error(
            "Model file not found: %s",
            model_path
        )

        raise

    except Exception as e:

        logger.error(
            "Failed to load models: %s",
            e
        )

        raise


# ==========================
# Model Evaluation
# ==========================

def model_evaluation(
    model,
    x_test,
    y_test,
    x_train,
    y_train,
    model_name
):

    try:

        # ==========================
        # Predictions
        # ==========================

        y_test_pred = model.predict(
            x_test
        )

        y_train_pred = model.predict(
            x_train
        )

        # ==========================
        # Metrics
        # ==========================

        result = {

            "train_accuracy": accuracy_score(
                y_train,
                y_train_pred
            ),

            "test_accuracy": accuracy_score(
                y_test,
                y_test_pred
            ),

            "precision": precision_score(
                y_test,
                y_test_pred,
                zero_division=0
            ),

            "recall": recall_score(
                y_test,
                y_test_pred,
                zero_division=0
            ),

            "f1_score": f1_score(
                y_test,
                y_test_pred,
                zero_division=0
            ),
        }

        # ==========================
        # Logging
        # ==========================

        logger.info(
            "============================================"
        )

        logger.info(
            "%s",
            model_name
        )

        logger.info(
            "============================================"
        )

        logger.info(
            "Train Accuracy : %.4f",
            result["train_accuracy"]
        )

        logger.info(
            "Test Accuracy  : %.4f",
            result["test_accuracy"]
        )

        logger.info(
            "Precision       : %.4f",
            result["precision"]
        )

        logger.info(
            "Recall          : %.4f",
            result["recall"]
        )

        logger.info(
            "F1 Score        : %.4f",
            result["f1_score"]
        )

        return result

    except Exception as e:

        logger.error(
            "Error during %s evaluation: %s",
            model_name,
            e
        )

        raise


# ==========================
# Save Metrics
# ==========================

def save_data(results):

    try:

        os.makedirs(
            "model_evaluation",
            exist_ok=True
        )

        file_path = os.path.join(
            "model_evaluation",
            "model_evaluation.txt"
        )

        with open(
            file_path,
            "w"
        ) as file:

            file.write(
                "MODEL EVALUATION RESULTS\n"
            )

            file.write(
                "============================================\n\n"
            )

            # ==========================
            # Random Forest Results
            # ==========================

            file.write(
                "RANDOM FOREST MODEL\n"
            )

            file.write(
                "--------------------------------------------\n"
            )

            for key, value in results[
                "random_forest"
            ].items():

                file.write(
                    f"{key}: {value:.4f}\n"
                )

            file.write("\n")

            # ==========================
            # Stacking Results
            # ==========================

            file.write(
                "STACKING CLASSIFIER\n"
            )

            file.write(
                "--------------------------------------------\n"
            )

            for key, value in results[
                "stacking"
            ].items():

                file.write(
                    f"{key}: {value:.4f}\n"
                )

        logger.info(
            "Evaluation metrics saved successfully."
        )

        logger.info(
            "Evaluation file: %s",
            file_path
        )

    except Exception as e:

        logger.error(
            "Error saving metrics: %s",
            e
        )

        raise


# ==========================
# Main Function
# ==========================

def main():

    try:

        logger.info(
            "============================================"
        )

        logger.info(
            "MODEL EVALUATION STARTED"
        )

        logger.info(
            "============================================"
        )

        # ==========================
        # Dataset Paths
        # ==========================

        test_path = (
            "data/processed/test.csv"
        )

        train_path = (
            "data/processed/train.csv"
        )

        # ==========================
        # Model Path
        # ==========================

        model_path = (
            "model_data/models.pkl"
        )

        # ==========================
        # Load Data
        # ==========================

        df_test, df_train = load_data(
            test_path,
            train_path
        )

        # ==========================
        # Separate Features / Target
        # ==========================

        target_column = "addicted_label"

        x_train = df_train.drop(
            target_column,
            axis=1
        )

        y_train = df_train[
            target_column
        ]

        x_test = df_test.drop(
            target_column,
            axis=1
        )

        y_test = df_test[
            target_column
        ]

        logger.info(
            "Target column: %s",
            target_column
        )

        logger.info(
            "X_train shape: %s",
            x_train.shape
        )

        logger.info(
            "X_test shape: %s",
            x_test.shape
        )

        # ==========================
        # Load Models
        # ==========================

        models = load_models(
            model_path
        )

        rf_model = models[
            "random_forest"
        ]

        stacking_model = models[
            "stacking"
        ]

        # ==========================
        # Evaluate Random Forest
        # ==========================

        rf_result = model_evaluation(
            rf_model,
            x_test,
            y_test,
            x_train,
            y_train,
            "RANDOM FOREST MODEL"
        )

        # ==========================
        # Evaluate Stacking
        # ==========================

        stacking_result = model_evaluation(
            stacking_model,
            x_test,
            y_test,
            x_train,
            y_train,
            "STACKING CLASSIFIER"
        )

        # ==========================
        # Store Results
        # ==========================

        results = {

            "random_forest": rf_result,

            "stacking": stacking_result

        }

        # ==========================
        # Save Results
        # ==========================

        save_data(
            results
        )

        # ==========================
        # Completed
        # ==========================

        logger.info(
            "============================================"
        )

        logger.info(
            "MODEL EVALUATION COMPLETED SUCCESSFULLY"
        )

        logger.info(
            "============================================"
        )

    except Exception as e:

        logger.error(
            "Error in main(): %s",
            e
        )

        raise


# ==========================
# Entry Point
# ==========================

if __name__ == "__main__":

    main()