import setuptools

setuptools.setup(
    name="RoboticWareHouse",
    version="0.0.1",
    author="Jonas & Johan",
    author_email="jonas@valfridsson.net",
    description="Simulator of a robotic warehouse",
    url="https://github.com/kex2019/robotic_warehouse",
    packages=["robotic_warehouse"],
    install_requires=["gym==0.10.5", "colorlog", "numpy==1.14.2"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
