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
    file_handler = logging.FileHandler("logs/model_evaluation.log")
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

        logger.info("Datasets loaded successfully.")
        return df_test, df_train

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

# ==========================
# Model Evaluation
# ==========================

def model_evaluation(model, x_test, y_test, x_train, y_train):
    try:
        y_test_pred = model.predict(x_test)
        y_train_pred = model.predict(x_train)

        result = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred, zero_division=0),
            "recall": recall_score(y_test, y_test_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
        }

        logger.info("Model evaluated successfully.")
        return result

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

# ==========================
# Save Metrics
# ==========================

def save_data(result):
    try:
        os.makedirs("model_evaluation", exist_ok=True)

        file_path = os.path.join(
            "model_evaluation",
            "model_evaluation.txt",
        )

        with open(file_path, "w") as file:
            for key, value in result.items():
                file.write(f"{key}: {value}\n")

        logger.info("Evaluation metrics saved successfully.")

    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise

# ==========================
# Main Function
# ==========================

def main():
    try:
        logger.info("Model evaluation started.")

        test_path = "data/processed/test.csv"
        train_path = "data/processed/train.csv"

        df_test, df_train = load_data(test_path, train_path)

        x_train = df_train.drop("addicted_label", axis=1)
        y_train = df_train["addicted_label"]

        x_test = df_test.drop("addicted_label", axis=1)
        y_test = df_test["addicted_label"]

        with open("model_data/models.pkl", "rb") as file:
            model = pickle.load(file)

        logger.info("Model loaded successfully.")

        result = model_evaluation(
            model,
            x_test,
            y_test,
            x_train,
            y_train,
        )

        save_data(result)

        logger.info("Model evaluation completed successfully.")

    except Exception as e:
        logger.error(f"Error in main(): {e}")
        raise

if __name__ == "__main__":
    main()