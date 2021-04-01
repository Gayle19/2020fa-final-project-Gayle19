import pandas as pd
from luigi import Task, Parameter, LocalTarget, ExternalTask
from machine_learning_utils.functions.functions import encode_onehot, impute_data
from machine_learning_utils.luigi.task import Requirement
from machine_learning_utils.luigi.data import DownloadData, DataPreprocess, BuildModel


class DownloadBreastCancerData(DownloadData):
    """Luigi task to demonstrate how to download the breast cancer data
    and store it in a data directory
    """

    # Fill in the DATA Parameter
    DATA = Parameter(default="~/datasets/module03_breast_cancer.csv")


class CleanBreastCancerData(DataPreprocess):
    """Luigi task to demonstrate how to clean the breast cancer data
    by dropping duplicates and fill na values
    """

    download = Requirement(DownloadBreastCancerData)

    def run(self):
        df = pd.read_parquet(self.input()["download"].path, engine="pyarrow")
        df["is_duplicate"] = df.duplicated()
        df = df.loc[df["is_duplicate"] == False]
        df.drop(columns="is_duplicate", inplace=True)

        # Replace '?' with mode - value/level with highest frequency in the feature

        df["node-caps"] = df["node-caps"].replace({"?": "no"})
        df["breast-quad"] = df["breast-quad"].replace({"?": "left_low"})

        # Fill in NA numerical values with mean

        impute_data(df, columns=list(df.columns))

        # Remove incorrect age values

        df = df.loc[(df["age"] != 250) & (df["age"] != -5)]
        df.reset_index(drop=True, inplace=True)

        df.to_parquet(self.output().path, engine="pyarrow")


class ExtractFeatures(DataPreprocess):
    """Luigi task to demonstrate how to extract relevant features
    from the breast cancer data
    """

    cleandata = Requirement(CleanBreastCancerData)

    def run(self):
        df = pd.read_parquet(self.input()["cleandata"].path, engine="pyarrow")

        # One hot encode categorical features
        df_o = encode_onehot(df, "menopause")
        df_o = encode_onehot(df_o, "node-caps")
        df_o = encode_onehot(df_o, "breast")
        df_o = encode_onehot(df_o, "breast-quad")
        df_o = encode_onehot(df_o, "irradiat")

        df_o.to_parquet(self.output().path, engine="pyarrow")


class BuildBCModel(BuildModel):
    """Luigi task to demonstrate how to build a model
    to correctly classify whether breast cancer is likely to reoccur.
    """

    preprocess = Requirement(ExtractFeatures)
    class_column = Parameter(default="recurrence")

