import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gym-trajectory",
    version="0.1",
    author="Daniel C. Kup",
    author_email="danielc@gmail.com",
    description="Custom environment for OpenAI gym",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DaKup/gym-trajectory",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['gym>=0.17.3', 'numpy']
)