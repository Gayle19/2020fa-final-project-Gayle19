import os
import shutil
import pandas as pd
import numpy as np
from luigi import Parameter, build
from unittest import TestCase
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from machine_learning_utils.functions.functions import eval_classifier, encode_onehot, impute_data
from machine_learning_utils.luigi.task import Requirement
from machine_learning_utils.luigi.data import DownloadData, DataPreprocess, BuildModel


def generate_data():
    """Function to help generate data for testing"""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


class MockDownloadData(DownloadData):
    DATA = Parameter(default="data/test_data.csv")


class MockDataPreprocess(DataPreprocess):
    download = Requirement(MockDownloadData)

    def run(self):
        df = pd.read_parquet(self.input()["download"].path, engine="pyarrow")
        df["is_duplicate"] = df.duplicated()
        df = df.loc[df["is_duplicate"] == False]
        df.drop(columns="is_duplicate", inplace=True)
        df.to_parquet(self.output().path, engine="pyarrow")


class MockBuildModel(BuildModel):
    preprocess = Requirement(MockDataPreprocess)
    class_column = Parameter(default="target")


class EvalClassifierTest(TestCase):
    def test_classifier(self):
        data = load_breast_cancer()
        X = data.data
        y = data.target
        df = eval_classifier(
            RandomForestClassifier(
                n_estimators=200, max_depth=5, random_state=None, n_jobs=4
            ),
            X,
            y,
        )
        self.assertEqual(df.shape[0], 4),
        self.assertEqual(df.shape[1], 5)


class EncodeOnehotTest(TestCase):
    def test_encode(self):
        data = {
            "animals": ["dog", "cat", "rabbit", "snake"],
            "numbers": [1, 2, 3, 4],
            "color": ["black", "white", "pink", "orange"],
        }
        df = pd.DataFrame(data)
        df2 = encode_onehot(df, "animals")

        # Test that the number of columns increase
        self.assertEqual(len(df2.columns), 6)

        # Test that the type of the new animals columns are integers
        self.assertEqual(df2["animals - dog"].dtype, "uint8")

        # Test that the original animal column does not exist
        with self.assertRaises(KeyError):
            df2["animals"]


class ImputeDataTest(TestCase):
    def test_impute(self):
        data = {
            "animals": ["dog", "cat", None, "dog"],
            "numbers": [1, 2, np.nan, 4],
            "color": ["black", "white", "pink", "orange"],
        }
        df = pd.DataFrame(data)
        impute_data(df, columns=["animals", "numbers"])

        # Check that na values have been removed
        self.assertEqual(df.isna().any().sum(), 0)

        # Check that na value in animals column have been replaced with correct value
        self.assertEqual(df["animals"][2], "dog")


class LuigiWorkflowTests(TestCase):
    filename = "test_data.csv"
    tmp_dir = "data/"

    def setUp(self):
        """Setup temporary directory for testing and monkey
        patch YelpReviews for local filesystem access.
        """

        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)

        # Output the test file:
        df = generate_data()
        test_file_path = os.path.join(self.tmp_dir, self.filename)
        df.to_csv(test_file_path)

    def tearDown(self):
        # Delete temporary directory and files:
        shutil.rmtree(self.tmp_dir)

    def test_download(self):
        task = MockDownloadData()

        build([task], local_scheduler=True)

        self.assertTrue(task.complete())

        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "data.parquet")))

    def test_preprocess(self):
        task = MockDataPreprocess()

        build([task], local_scheduler=True)

        self.assertTrue(task.complete())

        self.assertTrue(
            os.path.exists(os.path.join(self.tmp_dir, "MockDataPreprocess.parquet"))
        )

    def test_build(self):
        task = MockBuildModel()

        build([task], local_scheduler=True)

        self.assertTrue(task.complete())

        self.assertTrue(
            os.path.exists(os.path.join(self.tmp_dir, "MockBuildModel.csv"))
        )
