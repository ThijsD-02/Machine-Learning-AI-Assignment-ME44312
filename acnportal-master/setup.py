import setuptools

file_path = "C:\\Users\\alene\\Virtual_environments\\ME44312-2\\Machine-Learning-AI-Assignment-ME44312\\acnportal-master\\README.md"

with open(file_path, "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="acnportal",
    version="0.3.3",
    author="Zachary Lee, Sunash Sharma",
    author_email="ev-help@caltech.edu",
    url="https://github.com/zach401/acnportal",
    description="A package of tools for large-scale EV charging research.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={
        "": ["LICENSE.txt", "THANKS.txt",],
        "acnportal": ["signals/tariffs/tariff_schedules/*"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "requests",
        "pytz",
        "typing_extensions",
        "scikit-learn"
    ],
    extras_require={"all": ["scikit-learn"], "scikit-learn": ["scikit-learn"]},
)
