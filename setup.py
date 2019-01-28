import setuptools

setuptools.setup(
    name="RoboticWareHouse",
    version="0.0.1",
    author="Jonas & Johan",
    author_email="jonas@valfridsson.net",
    description="Simulator of a robotic warehouse",
    url="https://github.com/kex2019/robotic_warehouse",
    packages=["robotic_warehouse"],
    install_requires=["gym", "colorlog", "numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
