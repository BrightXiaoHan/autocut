from setuptools import setup, find_packages

requirements = [
    "srt",
    "moviepy",
    "opencc-python-reimplemented",
    "parameterized",
    "librosa",
    "torchaudio",
    "ctranslate2>=3.1.0",
    "transformers>=4.24.0",
]


setup(
    name="autocut",
    install_requires=requirements,
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "autocut = autocut.main:main",
        ]
    },
)
