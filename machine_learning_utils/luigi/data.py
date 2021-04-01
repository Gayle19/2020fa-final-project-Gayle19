import os
import pandas as pd
from luigi.format import Nop
from luigi import Task, Parameter, LocalTarget
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from machine_learning_utils.luigi.task import Requires, Requirement, TargetOutput
from machine_learning_utils.luigi.target import BaseAtomicProviderLocalTarget
from machine_learning_utils.functions.functions import eval_classifier



class DownloadData(Task):
    """A Luigi task to download data locally"""

    DATA = Parameter()  # Add in where the data should be downloaded
    LOCAL_ROOT = Parameter(default=os.path.abspath("data"))
    SHARED_RELATIVE_PATH = Parameter(default="data.parquet")

    def output(self):
        return BaseAtomicProviderLocalTarget(
            path=os.path.join(self.LOCAL_ROOT, self.SHARED_RELATIVE_PATH), format=Nop
        )

    def run(self):
        with self.output().open("wb") as out_file:
            df = pd.read_csv(self.DATA)
            df.to_parquet(out_file, engine="pyarrow")


class DataPreprocess(Task):
    """A Luigi task to preprocess data after download"""

    requires = Requires()
    # download = Requirement(DownloadData) - This task requires the DownloadData task as input
    output = TargetOutput(ext=".parquet", target_class=LocalTarget)

    def run(self):
        # Read in the data using df = pd.read_parquet(self.input()['download'].path, engine='pyarrow')
        # After you have implemented the necessary functions needed
        # extract the correct features, save data to parquet file using the statement below
        # df.to_parquet(self.output().path, engine='pyarrow')
        raise NotImplementedError()


class BuildModel(Task):
    """A Luigi task to classify a data using Support Vector Machine and Random Forest Model"""

    requires = Requires()
    preprocess = Requirement(DataPreprocess)
    class_column = Parameter()  # Add in the column that should be evaluated
    output = TargetOutput(ext=".csv", target_class=LocalTarget)

    def run(self):
        # Edit the classifiers to fit needs
        df = pd.read_parquet(self.input()["preprocess"].path, engine="pyarrow")
        X = df.loc[:, df.columns != self.class_column].values
        y = df.loc[:, df.columns == self.class_column].values.ravel()
        svc_report = eval_classifier(SVC(kernel="rbf", gamma=0.05, C=2), X, y)
        rf_report = eval_classifier(
            RandomForestClassifier(n_estimators=100, random_state=None, n_jobs=4), X, y
        )
        class_report = svc_report.join(rf_report, lsuffix="_svc", rsuffix="_rf")
        class_report.to_csv(self.output().path)
