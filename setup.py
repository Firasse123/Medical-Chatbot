from setuptools import setup, find_packages

setup(
    name="Medical_Chatbot",
    version="0.0.0",
    packages=find_packages(),  # look inside src/
    install_requires=[],
    author="Firas Guizani",
    author_email="firasguizani2012@gmail.com"
)

#It automatically finds all Python packages in the project starting from the same folder as setup.py.

#A folder is considered a package if it contains:__init__.py