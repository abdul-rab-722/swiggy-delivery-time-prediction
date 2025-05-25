import pandas as pd
import yaml
import logging
import joblib
from pathlib import Path
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor

TARGET = "time_taken"

#create logger
logger = logging.getLogger("model_trainig")
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


def read_params(params_path: Path) -> dict:
    try:
        with open(params_path, 'r') as file:
            params_file = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"File not found at : {params_path}")
    return params_file


def save_model(model, save_dir: Path, model_name: str) -> None:
    try:
        save_location = save_dir / model_name
        joblib.dump(value=model, filename=save_location)
        logger.info(f"Model saved at {save_location}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")


def save_transformer(transformer, save_dir: Path, transformer_name: str) -> None:
    # form the save location
    save_location = save_dir / transformer_name
    # save the transformer
    try:
        joblib.dump(value=transformer, filename=save_location)
        logger.info(f"Transformer saved successfully to {save_location}")
    except Exception as e:
        logger.error(f"Error saving transformer to {save_location}: {e}")


def train_model(model, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    try:
        model.fit(X_train, y_train)
        logger.info("Model training completed successfully.")
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise e
    return model


def make_X_and_y(data: pd.DataFrame, target_column: str) -> tuple:
    X = data.drop(columns=[target_column])
    y = data[target_column]
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y


if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    # train data path
    train_data_path = root_path / "data" / "processed" / "train_trans.csv"
    # parameters file
    params_file_path = root_path / "params.yaml"

    # load the trainig data
    training_data = load_data(train_data_path)
    logger.info(f"Training data loaded with shape: {training_data.shape}")

    # split the data into X and y
    X_train, y_train = make_X_and_y(training_data, TARGET)
    logger.info(f"Dataset splitting completed with X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # model parameters
    model_params = read_params(params_file_path)['Train']
    logger.info(f"Model parameters loaded: {model_params}")

    # rf_params
    rf_params = model_params['Random_Forest']
    logger.info(f"Random Forest parameters: {rf_params}")
    # build the Random Forest model
    rf = RandomForestRegressor(**rf_params)
    logger.info("Random Forest model created.")

    # lgbm_params
    lgbm_params = model_params['LightGBM']
    logger.info(f"LightGBM parameters: {lgbm_params}")
    # build the LightGBM model
    lgbm = LGBMRegressor(**lgbm_params)
    logger.info("LightGBM model created.")

    # meta model
    lr = LinearRegression()
    logger.info("Linear Regression model created for stacking.")

    # power transformer
    power_transformer = PowerTransformer()
    logger.info("Power Transformer created for target transformation.")

    # form stacking regressor
    stacking_regressor = StackingRegressor(estimators=[('rf_model', rf),
                                                   ('lgbm_model', lgbm)],
                                        final_estimator=lr,
                                        cv=5,
                                        n_jobs=-1)
    logger.info("Stacking Regressor created with Random Forest and LightGBM as base models.")

    # make the model wrapper
    model = TransformedTargetRegressor(regressor=stacking_regressor,
                                       transformer=power_transformer)
    logger.info("Transformed Target Regressor created with Stacking Regressor and Power Transformer.")

    # fit the model on training data
    train_model(model, X_train, y_train)
    logger.info("Model training completed successfully.")

    # model name
    model_filename = "model.joblib"
    # directory to save the model
    model_save_dir = root_path / "models"
    model_save_dir.mkdir(exist_ok=True)

    # extract the model from wrapper
    stacking_model = model.regressor_
    transformer = model.transformer_

    # save the model
    save_model(model=model, 
               save_dir=model_save_dir, 
               model_name=model_filename)
    logger.info(f"Model saved successfully at {model_save_dir / model_filename}")

    # save the stacking model
    stacking_model_filename = "stacking_model.joblib"
    save_model(model=stacking_model, 
               save_dir=model_save_dir, 
               model_name=stacking_model_filename)
    logger.info(f"Stacking model saved successfully at {model_save_dir / stacking_model_filename}")

    # save the transformer
    transformer_filename = "power_transformer.joblib"
    save_transformer(transformer=transformer, 
                     save_dir=model_save_dir, 
                     transformer_name=transformer_filename)
    logger.info(f"Transformer saved successfully at {model_save_dir / transformer_filename}")
    logger.info("Training script completed successfully.")