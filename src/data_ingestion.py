import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================
# LOGGING CONFIGURATION
# ============================================

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(
        os.path.join(log_dir, "data_ingestion.log")
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
# LOAD DATA
# ============================================

def load_data(data_url):
    try:
        df = pd.read_csv(data_url)

        logger.info("Dataset loaded successfully")
        logger.info("Dataset shape: %s", df.shape)

        return df

    except FileNotFoundError:
        logger.error("Dataset not found: %s", data_url)
        raise

    except pd.errors.ParserError as e:
        logger.error("Error parsing dataset: %s", e)
        raise

    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        raise


# ============================================
# SAVE TRAIN AND TEST DATA
# ============================================

def save_data(df_train, df_test):
    try:

        save_path = os.path.join("data", "raw")
        os.makedirs(save_path, exist_ok=True)

        train_path = os.path.join(save_path, "train.csv")
        test_path = os.path.join(save_path, "test.csv")

        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)

        logger.info("Training data saved to: %s", train_path)
        logger.info("Testing data saved to: %s", test_path)

    except Exception as e:
        logger.error("Failed to save train/test data: %s", e)
        raise


# ============================================
# MAIN
# ============================================

def main():

    try:

        # Dataset path
        data_url = r"C:\Users\hp\OneDrive\Desktop\git tutorial\project-1-MLOP\notebook\train.csv"

        # Train-test split parameters
        test_size = 0.2
        random_state = 42

        # Load dataset
        df = load_data(data_url)

        # Split dataset
        df_train, df_test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )

        logger.info(
            "Data split successfully: train=%s, test=%s",
            df_train.shape,
            df_test.shape
        )

        # Save train and test data
        save_data(df_train, df_test)

        logger.info("Data ingestion completed successfully!")

    except Exception as e:

        logger.error(
            "Data ingestion process failed: %s",
            e
        )

        raise


# ============================================
# ENTRY POINT
# ============================================

main()