import pandas as pd
import os
import logging
from sklearn.preprocessing import LabelEncoder

# ==========================
# Logging Configuration
# ==========================
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(
        os.path.join(log_dir, "feature_engineering.log")
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
# Load Data
# ==========================
def load_data(data_path):
    df = pd.read_csv(data_path)
    logger.debug(f"Loaded data from {data_path}")
    return df


# ==========================
# Feature Engineering
# ==========================
def create_features(df):

    df["recreational_screen_time"] = (
        df["social_media_hours"] + df["gaming_hours"]
    )

    df["other_screen_time"] = (
        df["daily_screen_time_hours"]
        - df["social_media_hours"]
        - df["gaming_hours"]
    )

    df["weekend_screen_difference"] = (
        df["weekend_screen_time"]
        - df["daily_screen_time_hours"]
    )

    df["weekend_screen_ratio"] = (
        df["weekend_screen_time"]
        / (df["daily_screen_time_hours"] + 1e-6)
    )

    df["screen_sleep_ratio"] = (
        df["daily_screen_time_hours"]
        / (df["sleep_hours"] + 1e-6)
    )

    df["social_media_ratio"] = (
        df["social_media_hours"]
        / (df["daily_screen_time_hours"] + 1e-6)
    )

    df["gaming_ratio"] = (
        df["gaming_hours"]
        / (df["daily_screen_time_hours"] + 1e-6)
    )

    df["app_opens_per_screen_hour"] = (
        df["app_opens_per_day"]
        / (df["daily_screen_time_hours"] + 1e-6)
    )

    df["notifications_per_screen_hour"] = (
        df["notifications_per_day"]
        / (df["daily_screen_time_hours"] + 1e-6)
    )

    df["screen_to_work_ratio"] = (
        df["daily_screen_time_hours"]
        / (df["work_study_hours"] + 1e-6)
    )

    df["total_tracked_hours"] = (
        df["daily_screen_time_hours"]
        + df["sleep_hours"]
        + df["work_study_hours"]
    )

    df["free_time_hours"] = (
        24 - df["sleep_hours"] - df["work_study_hours"]
    )

    df["screen_time_category"] = pd.cut(
        df["daily_screen_time_hours"],
        bins=[-float("inf"), 4, 8, float("inf")],
        labels=["Low", "Moderate", "High"],
    )

    df["sleep_category"] = pd.cut(
        df["sleep_hours"],
        bins=[-float("inf"), 6, 8, float("inf")],
        labels=["Low Sleep", "Normal Sleep", "High Sleep"],
    )

    df["high_screen_time"] = (
        df["daily_screen_time_hours"] >= 8
    ).astype(int)

    df["high_social_media_usage"] = (
        df["social_media_hours"] >= 4
    ).astype(int)

    df["high_gaming_usage"] = (
        df["gaming_hours"] >= 4
    ).astype(int)

    return df


def feature_engineering(train_df, test_df):
    try:
        train_df = create_features(train_df)
        test_df = create_features(test_df)

        categorical_cols = train_df.select_dtypes(
            include=["object", "category", "string"]
        ).columns

        logger.debug(f"Categorical Columns: {list(categorical_cols)}")

        for col in categorical_cols:
            le = LabelEncoder()

            train_df[col] = le.fit_transform(train_df[col].astype(str))

            mapping = {
                cls: idx for idx, cls in enumerate(le.classes_)
            }

            test_df[col] = (
                test_df[col]
                .astype(str)
                .map(mapping)
                .fillna(-1)
                .astype(int)
            )

        logger.debug("Feature engineering completed successfully.")
        return train_df, test_df

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise


# ==========================
# Save Data
# ==========================
def save_data(train_df, test_df):
    save_dir = os.path.join("data", "processed")
    os.makedirs(save_dir, exist_ok=True)

    train_df.to_csv(
        os.path.join(save_dir, "train.csv"),
        index=False,
    )

    test_df.to_csv(
        os.path.join(save_dir, "test.csv"),
        index=False,
    )

    logger.debug("Processed data saved successfully.")


# ==========================
# Main
# ==========================
def main():

    logger.info("Feature Engineering Started")

    train_df = load_data(
        r"C:\Users\hp\OneDrive\Desktop\git tutorial\project-1-MLOP\data\interim\train.csv"
    )

    test_df = load_data(
        r"C:\Users\hp\OneDrive\Desktop\git tutorial\project-1-MLOP\data\interim\test.csv"
    )

    train_df, test_df = feature_engineering(
        train_df,
        test_df,
    )

    save_data(train_df, test_df)

    logger.info("Feature Engineering Completed Successfully!")


main()