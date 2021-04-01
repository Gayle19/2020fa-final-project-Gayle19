[![Build Status](https://travis-ci.com/Gayle19/2020fa-final-project-Gayle19.svg?token=YFVNjWMfbzHWZG26kJpy&branch=master)](https://travis-ci.com/Gayle19/2020fa-final-project-Gayle19)

[![Maintainability](https://api.codeclimate.com/v1/badges/7e39f3d26760510b0a93/maintainability)](https://codeclimate.com/repos/5fda806aa3137a01a2006bcc/maintainability)

[![Test Coverage](https://api.codeclimate.com/v1/badges/7e39f3d26760510b0a93/test_coverage)](https://codeclimate.com/repos/5fda806aa3137a01a2006bcc/test_coverage)

# Machine Learning Utils

## Objective

This machine learning (ML) utilities package provides the necessary requirements needed to solve a binary supervised classification problem.
It creates a template that contains the basic steps for building a model (preprocessing, training/testing, and producing an outcome)
that can be restructured and expanded to fit other problems of similar type.

## Purpose
My motivation for creating ths package was to increase efficiency within my organization by developing a centralized location for all ML needs.
This package includes the tools and setup required to easily allow someone to implement the mudane task of ML so that now the focus can shift more towards improving model accuracy.
This way we can ensure that our models are being structured similarly and are reproducible. It can also serve as a tutorial for those who are new to machine learning and dynamic programming.

## Tools
This package is designed using three main tools.

1. [Cookiecutter](https://cookiecutter.readthedocs.io/en/latest/) - A Command Line Interface (CLI) tool to create an application boilerplate from a template.
It uses Jinja2 (a templating system) to replace and customize folder and file names as well as content.
 ** For this project I used cookiecutter pylibrary template **

2. [Luigi](https://luigi.readthedocs.io/en/stable/) - A python package developed at Spotify to build complex pipelines of batch jobs.
It handles dependency resolution, workflow management, and visualization.

3. [Scikit-Learn](https://scikit-learn.org/stable/) - A free software machine learning library for Python.
It features various classification, regression and clustering algorithms


<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Machine Learning Utils Module](#Machine Learning Utils Module)
    - [Examples](#Examples)
    - [Functions](#Functions)
    - [Luigi](#Luigi)
    - [Tests](#Tests)
- [Extras](#Extras)


<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## Machine Learning Utils Module

This module contains four python packages that serve as utilities needed to began solving a supervised classification problem.

### Examples
This package includes three examples for how this module can be used to produced an outcome.
Datasets used in examples can be found on Kaggle or Scikit-Learn.

1. Breast_cancer - Evaluates whether a breast tumor is likely to reoccur or not.
2. Mushroom - Evaluates whether a mushroom is edible or poisonous.
2. Titanic - Evaluates whether a passenger survived or not.

### Functions
This package includes helper functions to increase readability and reduce computations.

1. Eval_Classifier - Evaluates the classifier and produces a pandas dataframe displaying the classification report of the classifier.
2. Encode_Onehot - Onehot-encodes categorical features in dataframe
3. Impute_Data - Applies imputation to missing values

### Luigi
This package includes three luigi task that builds a pipeline for classifying the model

1. DownloadData - Downloads data from local repository.
2. DataPreprocess - Allows user to implement functions to clean data and extract features for model.
3. BuildModel - Classifies the data using a Random Forest model and returns a csv file displaying the results.

### Tests
This package uses the unittest library to test the functions and luigi tasks used within the module.


## Extras

* [Pipenv](https://docs.pipenv.org/) is a packaging tool for Python that
automatically creates and manages a virtualenv as well as adds/removes packages from your Pipfile as you install/uninstall packages.
It also generates Pipfile.lock, which is used to produce deterministic builds.

* [Setuptools](https://packaging.python.org/guides/distributing-packages-using-setuptools/) is a stable library designed to facilitate packaging Python projects.

* [Travis CI](https://docs.travis-ci.com/user/for-beginners/) is a continous integration platform that supports the development process
by automatically building and testing code changes.

* [Pytest](https://docs.pytest.org/en/stable/getting-started.html) is a framework that helps you build simple and scalable tests.


