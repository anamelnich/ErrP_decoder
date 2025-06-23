from setuptools import setup, find_packages

setup(
    name="ErrP_decoder",
    version="0.1.0",
    description="EEG ErrP decoder pipeline",
    author="Ana Melnichuk",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "pyyaml",
        "mne"
    ],
    entry_points={
        "console_scripts": [
            "ErrP-decoder = ErrP_decoder.cli:main"
        ]
    },
) 