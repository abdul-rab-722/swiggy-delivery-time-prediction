import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path
import logging

TARGET = "time_taken"

# create logger
logger = logging.getLogger("data_preparation")
logger.setLevel(logging.INFO)

# consol handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Add formatter to handler
handler.setFormatter(formatter)


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error(f"File not found at : {data_path}")
    return df


def split_data(data: pd.DataFrame, test_size: float, random_state: int) -> tuple:
    train_data, test_data = train_test_split(data,
                                                test_size=test_size,
                                                random_state=random_state)
    return train_data, test_data


def read_params(file_path: Path) -> dict:
    with open(file_path, 'r') as f:
        param_file = yaml.safe_load(f)
    return param_file


def save_data(data: pd.DataFrame, save_path: Path) -> None:
    try:
        data.to_csv(save_path, index=False)
        logger.info(f"Data saved successfully to {save_path}")
    except Exception as e:
        logger.error(f"Error saving data to {save_path}: {e}")


if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    # data load path
    data_load_path = root_path / "data" / "cleaned" / "swiggy_cleaned.csv"
    # data save directory
    data_save_dir = root_path / "data" / "interim"
    # create directory if not exists
    data_save_dir.mkdir(parents=True, exist_ok=True)
    # cleaned data file name
    train_filename = "train.csv"
    test_filename = "test.csv"
    # data save path
    save_train_path = data_save_dir / train_filename
    save_test_path = data_save_dir / test_filename
    # parameters file
    params_file_path = root_path / "params.yaml"

    # load the data
    df = load_data(data_load_path)
    logger.info("Data loaded successfully")

    # read parameters
    parameters = read_params(params_file_path)['Data_Preparation']
    test_size = parameters['test_size']
    random_state = parameters['random_state']
    logger.info(f"Parameters loaded successfully: test_size={test_size}, random_state={random_state}")


    # split into train and test data
    train_data, test_data = split_data(df, test_size=test_size, random_state=random_state)
    logger.info("Data split into train and test sets")

    # save the train and test data
    data_subset = [train_data, test_data]
    save_paths = [save_train_path, save_test_path]
    filename_list = [train_filename, test_filename]
    for filename, path, data in zip(filename_list, save_paths, data_subset):
        save_data(data=data, save_path=path)
        logger.info(f"{filename.replace(".csv", "")} data saved successfully to {path}")