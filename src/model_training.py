import pandas as pd
import os
import logging
import pickle
import yaml

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# ============================================
# LOGGING CONFIGURATION
# ============================================

logger = logging.getLogger("model_training")
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    os.makedirs("logs", exist_ok=True)

    file_handler = logging.FileHandler(
        "logs/model_training.log"
    )
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ============================================
# LOAD PARAMETERS
# ============================================

def load_params(path="params.yaml"):

    try:

        with open(path, "r") as file:
            params = yaml.safe_load(file)

        logger.info(
            "Parameters loaded successfully from %s",
            path
        )

        return params

    except FileNotFoundError:

        logger.error(
            "Parameters file not found: %s",
            path
        )

        raise

    except yaml.YAMLError as e:

        logger.error(
            "Error parsing YAML file: %s",
            e
        )

        raise

    except Exception as e:

        logger.error(
            "Error loading parameters: %s",
            e
        )

        raise


# ============================================
# LOAD DATA
# ============================================

def load_data(path):

    try:

        df = pd.read_csv(path)

        logger.info(
            "Data loaded successfully from %s",
            path
        )

        logger.info(
            "Dataset shape: %s",
            df.shape
        )

        return df

    except FileNotFoundError:

        logger.error(
            "Data file not found: %s",
            path
        )

        raise

    except pd.errors.ParserError as e:

        logger.error(
            "Error parsing CSV file: %s",
            e
        )

        raise

    except Exception as e:

        logger.error(
            "Error loading data: %s",
            e
        )

        raise


# ============================================
# RANDOM FOREST MODEL
# ============================================

def random_forest_model(
    X_train,
    X_test,
    y_train,
    y_test,
    params
):

    try:

        # ----------------------------------------
        # Get parameters from YAML
        # ----------------------------------------

        n_estimators = params["n_estimators"]
        max_depth = params["max_depth"]
        min_samples_leaf = params["min_samples_leaf"]

        logger.info(
            "Random Forest parameters:"
        )

        logger.info(
            "n_estimators: %s",
            n_estimators
        )

        logger.info(
            "max_depth: %s",
            max_depth
        )

        logger.info(
            "min_samples_leaf: %s",
            min_samples_leaf
        )

        # ----------------------------------------
        # Create Random Forest
        # ----------------------------------------

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )

        # ----------------------------------------
        # Train Model
        # ----------------------------------------

        logger.info(
            "Training Random Forest model..."
        )

        rf.fit(
            X_train,
            y_train
        )

        logger.info(
            "Random Forest training completed."
        )

        # ----------------------------------------
        # Prediction
        # ----------------------------------------

        y_pred_rf = rf.predict(
            X_test
        )

        # ----------------------------------------
        # Evaluation
        # ----------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred_rf
        )

        precision = precision_score(
            y_test,
            y_pred_rf,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred_rf,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred_rf,
            zero_division=0
        )

        # ----------------------------------------
        # Logging
        # ----------------------------------------

        logger.info(
            "============================================"
        )

        logger.info(
            "RANDOM FOREST MODEL"
        )

        logger.info(
            "============================================"
        )

        logger.info(
            "Accuracy : %.4f",
            accuracy
        )

        logger.info(
            "Precision: %.4f",
            precision
        )

        logger.info(
            "Recall   : %.4f",
            recall
        )

        logger.info(
            "F1 Score : %.4f",
            f1
        )

        return rf

    except Exception as e:

        logger.error(
            "Random Forest training failed: %s",
            e
        )

        raise


# ============================================
# STACKING CLASSIFIER
# ============================================

def stacking_model(
    X_train,
    X_test,
    y_train,
    y_test
):

    try:

        # ----------------------------------------
        # Base Estimators
        # ----------------------------------------

        estimators = [

            (
                "rf",

                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=3,
                    min_samples_leaf=6,
                    random_state=42,
                    n_jobs=-1
                )
            ),

            (
                "dt",

                DecisionTreeClassifier(
                    random_state=42
                )
            )
        ]

        # ----------------------------------------
        # Final Estimator
        # ----------------------------------------

        final_estimator = LogisticRegression(
            max_iter=1000
        )

        # ----------------------------------------
        # Create Stacking Classifier
        # ----------------------------------------

        stack_model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=-1
        )

        # ----------------------------------------
        # Train Stacking Model
        # ----------------------------------------

        logger.info(
            "Training Stacking Classifier..."
        )

        stack_model.fit(
            X_train,
            y_train
        )

        logger.info(
            "Stacking Classifier training completed."
        )

        # ----------------------------------------
        # Prediction
        # ----------------------------------------

        y_pred = stack_model.predict(
            X_test
        )

        # ----------------------------------------
        # Evaluation
        # ----------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        # ----------------------------------------
        # Logging
        # ----------------------------------------

        logger.info(
            "============================================"
        )

        logger.info(
            "STACKING CLASSIFIER"
        )

        logger.info(
            "============================================"
        )

        logger.info(
            "Accuracy : %.4f",
            accuracy
        )

        logger.info(
            "Precision: %.4f",
            precision
        )

        logger.info(
            "Recall   : %.4f",
            recall
        )

        logger.info(
            "F1 Score : %.4f",
            f1
        )

        return stack_model

    except Exception as e:

        logger.error(
            "Stacking model training failed: %s",
            e
        )

        raise


# ============================================
# SAVE BOTH MODELS
# ============================================

def save_models(
    rf_model,
    stack_model
):

    try:

        # ----------------------------------------
        # Create Model Directory
        # ----------------------------------------

        model_path = "model_data"

        os.makedirs(
            model_path,
            exist_ok=True
        )

        # ----------------------------------------
        # Model File
        # ----------------------------------------

        model_file = os.path.join(
            model_path,
            "models.pkl"
        )

        # ----------------------------------------
        # Store Both Models
        # ----------------------------------------

        models = {

            "random_forest": rf_model,

            "stacking": stack_model

        }

        # ----------------------------------------
        # Save Models
        # ----------------------------------------

        with open(
            model_file,
            "wb"
        ) as file:

            pickle.dump(
                models,
                file
            )

        logger.info(
            "============================================"
        )

        logger.info(
            "MODELS SAVED SUCCESSFULLY"
        )

        logger.info(
            "============================================"
        )

        logger.info(
            "Random Forest saved inside models.pkl"
        )

        logger.info(
            "Stacking model saved inside models.pkl"
        )

        logger.info(
            "Model file: %s",
            model_file
        )

    except Exception as e:

        logger.error(
            "Error while saving models: %s",
            e
        )

        raise


# ============================================
# MAIN FUNCTION
# ============================================

def main():

    try:

        logger.info(
            "============================================"
        )

        logger.info(
            "MODEL TRAINING PIPELINE STARTED"
        )

        logger.info(
            "============================================"
        )

        # ========================================
        # LOAD PARAMETERS
        # ========================================

        params = load_params()

        model_params = params["model_training"]

        # ========================================
        # LOAD TRAINING DATA
        # ========================================

        train_df = load_data(
            "data/processed/train.csv"
        )

        # ========================================
        # LOAD TESTING DATA
        # ========================================

        test_df = load_data(
            "data/processed/test.csv"
        )

        # ========================================
        # TARGET COLUMN
        # ========================================

        target_column = "addicted_label"

        # ========================================
        # SEPARATE FEATURES AND TARGET
        # ========================================

        X_train = train_df.drop(
            target_column,
            axis=1
        )

        y_train = train_df[
            target_column
        ]

        X_test = test_df.drop(
            target_column,
            axis=1
        )

        y_test = test_df[
            target_column
        ]

        logger.info(
            "Target column: %s",
            target_column
        )

        logger.info(
            "X_train shape: %s",
            X_train.shape
        )

        logger.info(
            "y_train shape: %s",
            y_train.shape
        )

        logger.info(
            "X_test shape: %s",
            X_test.shape
        )

        logger.info(
            "y_test shape: %s",
            y_test.shape
        )

        # ========================================
        # RANDOM FOREST
        # ========================================

        logger.info(
            "Starting Random Forest..."
        )

        rf_model = random_forest_model(
            X_train,
            X_test,
            y_train,
            y_test,
            model_params
        )

        # ========================================
        # STACKING CLASSIFIER
        # ========================================

        logger.info(
            "Starting Stacking Classifier..."
        )

        stack_model = stacking_model(
            X_train,
            X_test,
            y_train,
            y_test
        )

        # ========================================
        # SAVE BOTH MODELS
        # ========================================

        save_models(
            rf_model,
            stack_model
        )

        # ========================================
        # PIPELINE COMPLETED
        # ========================================

        logger.info(
            "============================================"
        )

        logger.info(
            "MODEL TRAINING PIPELINE COMPLETED"
        )

        logger.info(
            "============================================"
        )

    except Exception as e:

        logger.error(
            "Model training pipeline failed: %s",
            e
        )

        raise


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":

    main()