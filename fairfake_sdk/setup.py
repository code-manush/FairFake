from setuptools import setup, find_packages

setup(
    name="fairfake",
    version="1.0.0",
    description="Algorithmic Fairness Auditing and Demographic Certification SDK for Deepfake Forensics",
    author="FairFake Project Team",
    author_email="research@fairfake.org",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0"
    ],
    entry_points={
        "console_scripts": [
            "fairfake=fairfake.cli:main",
            "fairfake-cli=fairfake.cli:main"
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License"
    ]
)
