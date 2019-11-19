from distutils.core import setup

setup(
    name='AutoDiff',
    version='0.1dev',
    description='Auto Differentiation package developed for CS-207 [2019]',
    author='Group 24',
    packages=['AutoDiff',],
    license='MIT',
    long_description=open('README.md').read(),
    install_requires=['numpy'],
)