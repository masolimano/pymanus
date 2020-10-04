import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pymanus", # Replace with your own username
    version="0.0.2",
    author="Manuel Solimano",
    author_email="masolimano@uc.cl",
    description="Personal suite of tools for astronomy research",
    long_description=long_description,
    long_description_content_type="text/markdown",
#    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU LGPLv3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
