import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config import RANDOM_STATE, RAW_DATA_PATH, TARGET, TEST_SIZE


def load_data() -> pd.DataFrame:
    """
    - Read the CSV file and return DataFrame
    - Remove duplicate rows
    - Return the cleaned DataFrame
    """
    df = pd.read_csv(RAW_DATA_PATH)

    df = df.drop_duplicates()

    return df


def split_data(df: pd.DataFrame):
    """
    - Split the data into X and y
    - Split the data into training and testing sets
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
