# imports
from TaxiFareModel.data import get_data, clean_data, prepare_X_y , holdout
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from memoized_property import memoized_property
import mlflow
from  mlflow.tracking import MlflowClient

class Trainer():
    MLFLOW_URI = "https://mlflow.lewagon.co/"
    experiment_name = "[Ger] Berlin WolfSchew TaxiFareModel + 1"

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
        ])

        time_pipe = Pipeline([
        ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        preproc_pipe = ColumnTransformer([
        ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
        ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        self.pipeline = Pipeline([
        ('preproc', preproc_pipe),
        ('linear_model', LinearRegression())
        ])

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X,self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        
        self.mlflow_log_param('model', 'linear regression')
        self.mlflow_log_metric('rmse', rmse)

        return rmse


    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

if __name__ == "__main__":
    # get data
    df = get_data()

    # clean data
    df = clean_data(df)
    print(df.shape)

    # set X and y
    X, y = prepare_X_y(df)
    print(X.shape)
    print(y.shape)

    # hold out
    X_train, X_test, y_train, y_test = holdout(X,y)

    # train
    test = Trainer(X_train, y_train)
    test.run()

    # evaluate
    test.evaluate(X_test, y_test)