import pandas as pd
import logging
from pathlib import Path
import joblib
import mlflow
import dagshub
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_score

# initialize dagshub
dagshub.init(repo_owner="rababdul5786",
             repo_name="swiggy-delivery-time-prediction",
             mlflow=True)

# set the mlflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/rababdul5786/swiggy-delivery-time-prediction.mlflow")

# set mlflow experiment name
mlflow.set_experiment("DVC Pipeline")

TARGET = "time_taken"

#create logger
logger = logging.getLogger("model_evaluation")
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

def make_X_and_y(df: pd.DataFrame, target: str) -> tuple:
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def load_model(model_path: Path):
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        logger.error(f"Model file not found at : {model_path}")
    return model


if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    # train data path
    train_data_path = root_path / "data" / "processed" / "train_trans.csv"
    test_data_path = root_path / "data" / "processed" / "test_trans.csv"
    # model path
    model_path = root_path / "models" / "model.joblib"

    # load the trainig data
    training_data = load_data(train_data_path)
    logger.info(f"Training data loaded with shape: {training_data.shape}")
    # load the testing data
    testing_data = load_data(test_data_path)
    logger.info(f"Testing data loaded with shape: {testing_data.shape}")

    # split the data into X and y
    X_train, y_train = make_X_and_y(training_data, TARGET)
    logger.info(f"Dataset splitting completed with X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    X_test, y_test = make_X_and_y(testing_data, TARGET)
    logger.info(f"Dataset splitting completed with X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


    # load the model
    model = load_model(model_path)
    logger.info(f"Model loaded successfully from {model_path}")

    # get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    logger.info("Predictions made on training and testing data.")

    # calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    logger.info(f"Training MAE: {train_mae}, R2: {train_r2}")
    logger.info(f"Testing MAE: {test_mae}, R2: {test_r2}")

    # calculate cross-validation scores
    cv_scores = cross_val_score(model,
                                X_train,
                                y_train,
                                cv=5,
                                scoring='neg_mean_absolute_error',
                                n_jobs=-1)
    cv_mae = -cv_scores.mean()
    logger.info(f"Cross-validation MAE: {cv_mae}")

    # log with mlflow
    with mlflow.start_run():
        # set tags
        mlflow.set_tag("model", "Food Delivery Time Prediction")

        # log parameters
        mlflow.log_params(model.get_params())

        # log metrics
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("mean_cv_score", cv_mae)

        # log indivisual cv scores
        mlflow.log_metrics({f"CV {num}": score for num, score in enumerate(-cv_scores)})

        # mlflow dataset input datatype
        train_data_input = mlflow.data.from_pandas(training_data, targets=TARGET)
        test_data_input = mlflow.data.from_pandas(testing_data, targets=TARGET)

        # log input datasets
        mlflow.log_input(dataset=train_data_input, context="training")
        mlflow.log_input(dataset=test_data_input, context="validation")
        logger.info("Input datasets logged to MLflow.")

        # log signature
        model_signature = mlflow.models.infer_signature(model_input=X_train.sample(20, random_state=42),
                                                        model_output=model.predict(X_train.sample(20, random_state=42)))
        logger.info("Model signature inferred successfully.")
        
        # log the stacking regressor model
        mlflow.sklearn.log_model(model, "model", signature=model_signature)
        logger.info("Model logged to MLflow successfully and MLflow logging completed.")