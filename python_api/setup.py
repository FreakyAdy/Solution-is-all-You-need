import setuptools

setuptools.setup(
    name="phantom_core",
    version="0.1.0",
    author="PHANTOM CORE Project",
    description="Python API for PHANTOM CORE Engine",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "msgpack",
        "torch>=2.0.0",
        "transformers",
    ],
    entry_points={
        "console_scripts": [
            "phantom-api=phantom.api.server:main",
        ],
    },
)
