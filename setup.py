from setuptools import setup, find_packages

from PWF_reconstruction import __version__

setup(
    name='PWF_reconstruction',
    version=__version__,

    url='https://github.com/arsenefer/PWF_reconstruction',
    author='Arsène Ferrière',
    description='Package to conduce PWF reconstruction and uncertainty estimations',
    author_email='arsene.ferriere@cea.fr',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "seaborn",
        "matplotlib",
        "scipy",
        "numdifftools"
    ]
)