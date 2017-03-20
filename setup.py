from setuptools import setup, find_packages

setup(
    name='dwtviz',
    version='0.0.8',
    description='Creates visualizations for discrete wavelet transforms',
    url='https://github.com/n-s-f/dwtviz',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['pywavelets', 'matplotlib'],
)
