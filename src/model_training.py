import pandas as pd
import os
import logging
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# ==========================
# Logging Configuration
# ==========================

logger = logging.getLogger("model_training")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler("logs/model_training.log")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ==========================
# Load Data
# ==========================

def load_data(path):
    try:
        df = pd.read_csv(path)
        logger.info("Data loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# ==========================
# Random Forest Model
# ==========================

def random_forest_model(X_train, X_test, y_train, y_test):

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    logger.info("----- Random Forest -----")
    logger.info(f"Accuracy : {accuracy_score(y_test, y_pred_rf)}")
    logger.info(f"Precision: {precision_score(y_test, y_pred_rf, zero_division=0)}")
    logger.info(f"Recall   : {recall_score(y_test, y_pred_rf, zero_division=0)}")
    logger.info(f"F1 Score : {f1_score(y_test, y_pred_rf, zero_division=0)}")

    return rf

# ==========================
# Stacking Classifier
# ==========================

def stacking_model(X_train, X_test, y_train, y_test):

    estimators = [
        ("rf", RandomForestClassifier()),
        ("dt", DecisionTreeClassifier())
    ]

    stack_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression()
    )

    stack_model.fit(X_train, y_train)
    y_pred = stack_model.predict(X_test)

    logger.info("----- Stacking Classifier -----")
    logger.info(f"Accuracy : {accuracy_score(y_test, y_pred)}")
    logger.info(f"Precision: {precision_score(y_test, y_pred, zero_division=0)}")
    logger.info(f"Recall   : {recall_score(y_test, y_pred, zero_division=0)}")
    logger.info(f"F1 Score : {f1_score(y_test, y_pred, zero_division=0)}")

    return stack_model

# ==========================
# Save Model
# ==========================

def save_data(model):
    try:
        path_model = "model_data"
        os.makedirs(path_model, exist_ok=True)

        path_model_file = os.path.join(path_model, "models.pkl")

        with open(path_model_file, "wb") as file:
            pickle.dump(model, file)

        logger.info(f"Model saved successfully at {path_model_file}")

    except Exception as e:
        logger.error(f"Error while saving model: {e}")
        raise

# ==========================
# Main Function
# ==========================

def main():
    try:
        train_df = load_data("data/processed/train.csv")
        test_df = load_data("data/processed/test.csv")

        X_train = train_df.drop("addicted_label", axis=1)
        y_train = train_df["addicted_label"]

        X_test = test_df.drop("addicted_label", axis=1)
        y_test = test_df["addicted_label"]

        rf_model = random_forest_model(X_train, X_test, y_train, y_test)

        stack_model = stacking_model(X_train, X_test, y_train, y_test)

        save_data(stack_model)

        logger.info("Model training pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()