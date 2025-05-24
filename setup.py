from setuptools import setup, find_packages

setup(
    name            ='AHPI',
    version         ='1.0.0',
    description     ='AHPI ranking algorithm Python package',
    author          ='Alexandre Mojon',
    author_email    ='alexandre.mojon@unisg.ch',
    packages        =find_packages(),
    install_requires=[          # Add runtime dependencies here
        'pandas',
        'numpy',
        'scipy'
    ],
    python_requires='>=3.10',
)