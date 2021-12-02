import pathlib

import setuptools


def read(HERE: pathlib.Path, filename, variable):
    namespace = {}

    exec(open(HERE / "torchdatasets" / filename).read(), namespace)  # get version
    return namespace[variable]


HERE = pathlib.Path(__file__).resolve().parent

setuptools.setup(
    name=read(HERE, pathlib.Path("_name.py"), "_name"),
    version=read(HERE, pathlib.Path("_version.py"), "__version__"),
    license="MIT",
    author="Szymon Maszke",
    author_email="szymon.maszke@protonmail.com",
    description="PyTorch based library focused on data processing and input pipelines in general.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/szymonmaszke/torchdatasets",
    packages=setuptools.find_packages(),
    install_requires=open("environments/requirements.txt").read().splitlines(),
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    project_urls={
        "Website": "https://szymonmaszke.github.io/torchdatasets",
        "Documentation": "https://szymonmaszke.github.io/torchdatasets/#torchdatasets",
        "Issues": "https://github.com/szymonmaszke/torchdatasets/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc",
    },
    keywords="pytorch torch data datasets map cache memory disk apply database",
)
