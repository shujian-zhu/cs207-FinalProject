import setuptools

setuptools.setup(
    name='AutoDiff-CS207-24',
    version='0.2',
    description='Auto Differentiation package developed for CS-207 [2019]',
    author='Group 24',
    packages=['AutoDiff',],
    license='MIT',
    long_description=open('README.md').read(),
    url="https://github.com/we-the-diff/cs207-FinalProject",
    install_requires=['numpy'],
    python_requires='>=3.6',
)