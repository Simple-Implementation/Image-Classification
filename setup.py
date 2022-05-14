from setuptools import setup, find_packages

setup(
    name="image_classification",
    description="Simple Image Classification",
    # url="https://github.com/Simple-Implementation/Image-Classification",
    python_requires='>=3',
    packages=find_packages(),
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
)