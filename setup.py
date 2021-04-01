#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


setup(
    name='machine-learning-utils',
    use_scm_version={
        'local_scheme': 'dirty-tag',
        'write_to': 'machine_learning_utils/_version.py',
        'fallback_version': '0.0.0',
        'version_scheme': 'release-branch-semver',
    },
    description='"A machine learning utilities package"',
    long_description='%s\n%s' % (
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))
    ),
    author='Gayle Turner',
    author_email='gat053@g.harvard.edu',
    url='https://github.com/Gayle19/2020fa-machinelearning-utils-Gayle19',
    packages=find_packages('machine_learning_utils'),
    package_dir={'': 'machine_learning_utils'},
    py_modules=[splitext(basename(path))[0] for path in glob('machine_learning_utils/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        # uncomment if you test on these interpreters:
        # 'Programming Language :: Python :: Implementation :: IronPython',
        # 'Programming Language :: Python :: Implementation :: Jython',
        # 'Programming Language :: Python :: Implementation :: Stackless',
        'Topic :: Utilities',
        'Private :: Do Not Upload',
    ],
    project_urls={
        'Changelog': 'https://github.com/Gayle19/2020fa-machinelearning-utils-Gayle19/blob/master/CHANGELOG.rst',
        'Issue Tracker': 'https://github.com/Gayle19/2020fa-machinelearning-utils-Gayle19/issues',
    },
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires='>=3.6',
    install_requires=[
        'luigi',
        'pandas',
        'numpy',
        'sklearn',
    ],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    setup_requires=[
        'setuptools_scm>=3.3.1',
    ],
    entry_points={
        'console_scripts': [
            'machine-learning-utils = machine_learning_utils.cli:main',
        ]
    },
)
