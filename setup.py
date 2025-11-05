from pathlib import Path

from setuptools import find_packages, setup

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="podcastpile",
    version="0.1.7",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "podcastpile.nisqa": ["weights/*.tar"],
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        # Manager dependencies
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "sqlalchemy>=2.0.0",
        "alembic>=1.12.0",
        "pydantic>=2.5.0",
        "python-multipart>=0.0.6",
        "jinja2>=3.1.2",
        "httpx>=0.25.0",
        "python-dateutil>=2.8.2",
        "click>=8.1.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        # Worker dependencies (heavy ML packages)
        "worker": [
            "librosa>=0.10.0",
            "soundfile>=0.12.0",
            "nemo_toolkit[asr]>=1.20.0",
            "megatron-core",
        ],
        # Development dependencies
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ppcli=podcastpile.cli:main",
        ],
    },
    python_requires=">=3.9",
)
