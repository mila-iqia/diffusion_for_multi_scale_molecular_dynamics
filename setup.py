from setuptools import find_packages, setup

setup(
    name='crystal_diffusion',
    version='0.0.1',
    packages=find_packages(include=['crystal_diffusion', 'crystal_diffusion.*']),
    python_requires='>=3.11',
    install_requires=[
        'flake8==4.0.1',
        'flake8-docstrings==1.6.0',
        'gitpython==3.1.27',
        'jupyter==1.0.0',
        'jinja2==3.1.2',
        'myst-parser==2.0.0',
        'orion>=0.2.4.post1',
        'pyyaml==6.0',
        'pytest==7.1.2',
        'pytest-cov==3.0.0',
        'pytorch_lightning>=2.2.0',
        'pytype==2024.2.13',
        'sphinx==7.2.6',
        'sphinx-autoapi==3.0.0',
        'sphinx-rtd-theme==2.0.0',
        'sphinxcontrib-napoleon==0.7',
        'sphinxcontrib-katex==0.8.6',
        'tensorboard==2.16.2',
        'tqdm==4.64.0',
        'torch==2.2.0',
        'torchvision>=0.17.0',
    ],
    entry_points={
        'console_scripts': [
            'cd-train=crystal_diffusion.train:main',
            'cd-eval=crystal_diffusion.evaluate:main',
        ],
    }
)
