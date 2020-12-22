import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pymanus",
    version="0.0.7",
    author="Manuel Solimano",
    author_email="masolimano at uc dot cl",
    description="Personal suite of tools for astronomy research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU LGPLv3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
