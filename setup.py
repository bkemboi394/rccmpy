
# -*- coding: utf-8 -*-
import setuptools

# Package meta-data.
NAME = 'rccmpy'
DESCRIPTION = 'A library for building a Random Covariance Clustering Model.'
URL = 'https://github.com/bkemboi394/rccmpy'
EMAIL = 'kemboib@mail.gvsu.edu'
AUTHOR = 'Beatrice Kemboi'
REQUIRES_PYTHON = '>=3.9.0'
VERSION = '0.1.1'


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# Other required third-party packages
REQUIRED = [
    'numpy', 'scipy'
]

# optional packages
EXTRAS = {
    # 'fancy feature': ['Django],
}



setuptools.setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    py_modules=['rccm'],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT'
)