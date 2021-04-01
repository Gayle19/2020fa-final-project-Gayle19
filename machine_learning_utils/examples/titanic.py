import re
import pandas as pd
from luigi import Task, Parameter, LocalTarget, ExternalTask
from machine_learning_utils.functions.functions import encode_onehot, impute_data
from machine_learning_utils.luigi.task import Requirement
from machine_learning_utils.luigi.data import DownloadData, DataPreprocess, BuildModel


class DownloadTitanicData(DownloadData):
    """Luigi task to demonstrate how to download the titanic dataset
    and store it in a data directory
    """

    DATA = Parameter(default="~/datasets/titanic.csv")


class CleanTitanicData(DataPreprocess):
    """Luigi task to demonstrate how to clean the titanic data
    by dropping duplicates and filling in na values
    """

    download = Requirement(DownloadTitanicData)

    def run(self):
        df = pd.read_parquet(self.input()["download"].path, engine="pyarrow")
        df["is_duplicate"] = df.duplicated()
        df = df.loc[df["is_duplicate"] == False]
        df.drop(columns=["is_duplicate", "boat", "body", "home.dest"], inplace=True)

        # Fill in NA values with mean
        impute_data(df, columns=list(df.columns))

        df.to_parquet(self.output().path, engine="pyarrow")


class ExtractFeatures(DataPreprocess):
    """Luigi task to demonstrate how to preprocess features
    in the titanic dataset
    """

    cleandata = Requirement(CleanTitanicData)

    def run(self):
        df = pd.read_parquet(self.input()["cleandata"].path, engine="pyarrow")

        # determine family size
        df["familysize"] = df["parch"] + df["sibsp"] + 1
        df["singleton"] = df["familysize"].map(lambda s: 1 if s == 1 else 0)
        df["smallfamily"] = df["familysize"].map(lambda s: 1 if 2 <= s <= 4 else 0)
        df["largefamily"] = df["familysize"].map(lambda s: 1 if 5 <= s else 0)

        # strip titles from names
        df["title"] = df["name"].map(
            lambda name: name.split(",")[1].split(".")[0].strip()
        )
        df["title"] = df["title"].replace(
            [
                "Lady",
                "Countess",
                "Capt",
                "Col",
                "Don",
                "Dr",
                "Major",
                "Rev",
                "Sir",
                "Jonkheer",
                "Dona",
            ],
            "Rare",
        )
        df["title"] = df["title"].replace("Mlle", "Miss")
        df["title"] = df["title"].replace("Ms", "Miss")
        df["title"] = df["title"].replace("Mme", "Mrs")

        # convert cabin into numbers
        df["deck"] = df["cabin"].map(
            lambda x: re.compile("([a-zA-Z]+)").search(x).group()
        )

        # onehot-encode nominal features
        df_o = encode_onehot(df, "embarked")
        df_o = encode_onehot(df_o, "sex")
        df_o = encode_onehot(df_o, "title")
        df_o = encode_onehot(df_o, "deck")

        df_o = df_o.drop(columns=["passenger_id", "name", "ticket", "cabin"])
        df_o.to_parquet(self.output().path, engine="pyarrow")


class BuildTModel(BuildModel):
    """Luigi task to demonstrate how to build a model
    to correctly classify whether a passenger on the
    Titanic survived or not.
    """

    preprocess = Requirement(ExtractFeatures)
    class_column = Parameter(default="survived")
