import os
from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    Reads requirements.txt safely using absolute path
    and returns a list of dependencies.
    """
    requirements = []
    
    # Convert to absolute path to avoid file not found errors
    file_path = os.path.abspath(file_path)

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]  # remove \n

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    print(requirements)
    return requirements


#  ✅ OPTIONAL: print once to verify (remove later)
# print(get_requirements("requirements.txt"))

setup(
    name="Heart Disease Prediction",
    version="0.0.1",
    author="Muneeb Mughal",
    author_email="muneerahmed8556@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
