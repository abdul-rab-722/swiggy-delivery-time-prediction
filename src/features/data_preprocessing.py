import pandas as pd
import logging
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler,
    OrdinalEncoder
)
import joblib
from sklearn import set_config

# set the transformer outputs to be pandas DataFrame
set_config(transform_output="pandas")

# columns to preprocess in data
num_cols = ["age",
            "ratings",
            "pickup_time_minutes",
            "distance"]
nominal_cat_cols = ['weather',
                    'type_of_order',
                    'type_of_vehicle',
                    "festival",
                    "city_type",
                    "is_weekend",
                    "order_time_of_day"]

ordinal_cat_cols = ["traffic","distance_type"]

target_col = "time_taken"

# generate order for ordinal encoding
traffic_order = ["low", "medium", "high", "jam"]
distance_type_order = ["short", "medium", "long", "very_long"]

#create logger
logger = logging.getLogger("data_preprocessing")
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


def drop_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"The original dataset with missing values has {data.shape[0]} rows and {data.shape[1]} columns.")
    df_dropped = data.dropna()
    logger.info(f"The dataset after dropping missing values has {df_dropped.shape[0]} rows and {df_dropped.shape[1]} columns.")
    missing_vals = df_dropped.isnull().sum().sum()

    if missing_vals > 0:
        raise ValueError(f"There are still {missing_vals} missing values in the dataset after dropping.")
    else:
        logger.info("No missing values found in the dataset after dropping.")
    return df_dropped

def save_transformer(transformer, save_dir: Path, transformer_name: str) -> None:
    # form the save location
    save_location = save_dir / transformer_name
    # save the transformer
    try:
        joblib.dump(value=transformer, filename=save_location)
        logger.info(f"Transformer saved successfully to {save_location}")
    except Exception as e:
        logger.error(f"Error saving transformer to {save_location}: {e}")


def train_preprocessor(preprocessor, data: pd.DataFrame):
    # fit the data
    preprocessor.fit(data)
    return preprocessor


def perform_transformations(preprocessor, data: pd.DataFrame) -> pd.DataFrame:
    # transform the data
    transformed_data = preprocessor.transform(data)
    return transformed_data


def save_data(data: pd.DataFrame, save_path: Path) -> None:
    data.to_csv(save_path, index=False)


def make_X_and_y(data: pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def join_X_and_y(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    # join based on indexes
    joined_df = X.join(y, how="inner")
    return joined_df


if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    # data load path
    train_data_path = root_path / "data" / "interim" / "train.csv"
    test_data_path = root_path / "data" / "interim" / "test.csv"
    # data save directory
    save_data_dir = root_path / "data" / "processed"
    # create directory if not exists
    save_data_dir.mkdir(parents=True, exist_ok=True)
    # cleaned data file name
    train_trans_filename = "train_trans.csv"
    test_trans_filename = "test_trans.csv"
    # data save path
    save_train_trans_path = save_data_dir / train_trans_filename
    save_test_trans_path = save_data_dir / test_trans_filename
    
    
    # preprocessor
    preprocessor = ColumnTransformer(transformers=[
            ("scale", MinMaxScaler(), num_cols),
            ("nominal_encode", OneHotEncoder(drop="first",
                                             handle_unknown="ignore",
                                             sparse_output=False), nominal_cat_cols),
            ("ordinal_encode", OrdinalEncoder(categories=[traffic_order,
                                                          distance_type_order],
                                            encoded_missing_value=-999,
                                            handle_unknown="use_encoded_value",
                                            unknown_value=-1), ordinal_cat_cols)
                                    ],remainder="passthrough",
                                    n_jobs=-1,
                                    force_int_remainder_cols=False,
                                    verbose_feature_names_out=False)

    # load the train and test data with missing values dropped
    train_df = drop_missing_values(load_data(data_path=train_data_path))
    logger.info("Train data loaded successfully")
    test_df = drop_missing_values(load_data(data_path=test_data_path))
    logger.info("Test data loaded successfully")


    # split the train and test data
    X_train, y_train = make_X_and_y(data=train_df, target_column=target_col)
    X_test, y_test = make_X_and_y(data=test_df, target_column=target_col)
    logger.info("Data splitting Completed")


    # fit the preprocessor on the X_train
    train_preprocessor(preprocessor=preprocessor, data=X_train)
    logger.info("preprocessor is trained")


    # transform the data
    X_train_trans = perform_transformations(preprocessor=preprocessor, data=X_train)
    logger.info("X_train data is transformed")
    X_test_trans = perform_transformations(preprocessor=preprocessor, data=X_test)
    logger.info("X_test data is transformed")


    # join back X and y
    train_trans_df = join_X_and_y(X=X_train_trans, y=y_train)
    test_trans_df = join_X_and_y(X=X_test_trans, y=y_test)
    logger.info("Datasets Joined Back")


    # save the transformed data
    data_subset = [train_trans_df, test_trans_df]
    data_paths = [save_train_trans_path, save_test_trans_path]
    filenames = [train_trans_filename, test_trans_filename]
    for filename, path, data in zip(filenames, data_paths, data_subset):
        save_data(data=data, save_path=path)
        logger.info(f"{filename.replace(".csv", "")} data saved to {path} location.")


    # save the preprocessor
    transformer_filename = "preprocessor.joblib"
    # directory to save transformer
    transformer_save_dir = root_path / "models"
    transformer_save_dir.mkdir(exist_ok=True)
    # save the transformer
    save_transformer(transformer=preprocessor, 
                    save_dir=transformer_save_dir, 
                    transformer_name=transformer_filename)