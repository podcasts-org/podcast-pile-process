from setuptools import find_packages, setup

setup(
    name="podcastpile",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
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
