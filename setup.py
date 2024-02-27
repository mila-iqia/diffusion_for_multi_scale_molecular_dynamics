from setuptools import find_packages, setup

with open('requirements.txt', 'r') as f:
    requirements = f.readlines()

setup(
    name='crystal_diffusion',
    version='0.0.1',
    packages=find_packages(include=['crystal_diffusion', 'crystal_diffusion.*']),
    python_requires='>=3.11',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'cd-train=crystal_diffusion.train:main',
            'cd-eval=crystal_diffusion.evaluate:main',
        ],
    }
)
