import setuptools

name, version = open("METADATA").read().splitlines()

setuptools.setup(
    name=name,
    version=version,
    license="MIT",
    author="Szymon Maszke",
    author_email="szymon.maszke@protonmail.com",
    description="PyTorch based library focused on data processing and input pipelines in general.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/torchdata",
    packages=setuptools.find_packages(),
    install_requires=open("environments/requirements.txt").read().splitlines(),
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
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
        "Website": "https://szymonmaszke.github.io/torchdata",
        "Documentation": "https://szymonmaszke.github.io/torchdata/#torchdata",
        "Issues": "https://github.com/szymonmaszke/torchdata/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc",
    },
    keywords="pytorch torch data datasets map cache memory disk apply database",
)
