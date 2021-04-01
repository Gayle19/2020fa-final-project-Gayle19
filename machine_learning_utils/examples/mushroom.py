import pandas as pd
from luigi import Task, Parameter, LocalTarget, ExternalTask
from machine_learning_utils.luigi.task import Requirement
from machine_learning_utils.luigi.data import DownloadData, DataPreprocess, BuildModel


class DownloadMushroomData(DownloadData):
    """Luigi task to demonstrate how to download the mushroom dataset
    and store it in a data directory
    """

    DATA = Parameter(default="~/datasets/Mushroom_Dataset.csv")


class CleanMushroomData(DataPreprocess):
    """Luigi task to demonstrate how to clean the mushroom data
    by dropping duplicates
    """

    download = Requirement(DownloadMushroomData)

    def run(self):
        df = pd.read_parquet(self.input()["download"].path, engine="pyarrow")
        df["is_duplicate"] = df.duplicated()
        df = df.loc[df["is_duplicate"] == False]

        df.drop(columns="is_duplicate", inplace=True)
        df.to_parquet(self.output().path, engine="pyarrow")


class ExtractFeatures(DataPreprocess):
    """Luigi task to demonstrate how to preprocess features
    in the mushroom dataset
    """

    cleandata = Requirement(CleanMushroomData)

    def run(self):
        df = pd.read_parquet(self.input()["cleandata"].path, engine="pyarrow")
        df_o = pd.get_dummies(df.iloc[:, 1:], columns=df.columns[1:])
        df_o["class"] = df["class"]
        df_o.to_parquet(self.output().path, engine="pyarrow")


class BuildMRModel(BuildModel):
    """Luigi task to demonstrate how to build a model
    to correctly classify whether a mushroom is edible or not.
    """

    preprocess = Requirement(ExtractFeatures)
    class_column = Parameter(default="class")
