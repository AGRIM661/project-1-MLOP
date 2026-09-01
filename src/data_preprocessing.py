import pandas as pd
import os
import logging


# ==========================
# Logging Configuration
# ==========================

logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# File Handler
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

file_handler = logging.FileHandler(
    os.path.join(log_dir, "data_preprocessing.log")
)
file_handler.setLevel(logging.DEBUG)

# Formatter
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

def load_data(data_path):
    try:
        df = pd.read_csv(data_path)

        logger.debug(f"Loaded data from {data_path}")
        logger.debug(f"Shape of data: {df.shape}")

        return df

    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        raise

    except pd.errors.ParserError as e:
        logger.error(f"Error parsing file: {e}")
        raise

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


# ==========================
# Data Preprocessing
# ==========================

def preprocessing_data(train_df, test_df):

    try:
        logger.debug("Starting data preprocessing...")

        # --------------------------
        # Remove ID column
        # --------------------------

        if "id" in train_df.columns:
            train_df.drop(columns=["id"], inplace=True)

        if "id" in test_df.columns:
            test_df.drop(columns=["id"], inplace=True)

        # --------------------------
        # Numerical Columns
        # --------------------------

        num_cols = train_df.select_dtypes(
            include="number"
        ).columns

        for col in num_cols:

            median_value = train_df[col].median()

            train_df[col] = train_df[col].fillna(median_value)
            test_df[col] = test_df[col].fillna(median_value)

        # --------------------------
        # Categorical Columns
        # --------------------------

        cat_cols = train_df.select_dtypes(
            include="object"
        ).columns

        for col in cat_cols:

            mode_value = train_df[col].mode()[0]

            train_df[col] = train_df[col].fillna(mode_value)
            test_df[col] = test_df[col].fillna(mode_value)

        logger.debug("Missing values handled successfully.")
        logger.debug("Data preprocessing completed.")

        return train_df, test_df

    except Exception as e:
        logger.error(f"Failed to preprocess data: {e}")
        raise


# ==========================
# Save Data
# ==========================

def save_data(train_df, test_df):

    try:

        save_dir = os.path.join(
            "data",
            "interim"
        )

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        train_path = os.path.join(
            save_dir,
            "train.csv"
        )

        test_path = os.path.join(
            save_dir,
            "test.csv"
        )

        train_df.to_csv(
            train_path,
            index=False
        )

        test_df.to_csv(
            test_path,
            index=False
        )

        logger.debug(
            f"Train data saved to: {train_path}"
        )

        logger.debug(
            f"Test data saved to: {test_path}"
        )

    except Exception as e:

        logger.error(
            f"Failed to save data: {e}"
        )

        raise


# ==========================
# Main Function
# ==========================

def main():

    try:

        train_path =r"data\raw\train.csv"


        test_path = r"data\raw\test.csv"

        # Load train data
        train_df = load_data(train_path)

        # Load test data
        test_df = load_data(test_path)

        # Preprocess data
        train_df, test_df = preprocessing_data(
            train_df,
            test_df
        )

        # Save processed data
        save_data(
            train_df,
            test_df
        )

        logger.info(
            "Data preprocessing pipeline completed successfully."
        )

    except Exception:

        logger.exception(
            "Data preprocessing pipeline failed."
        )

        raise


# ==========================
# Run Script
# ==========================

main()