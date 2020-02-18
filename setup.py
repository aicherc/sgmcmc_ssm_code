from setuptools import setup
config = {
    'name': 'sgmcmc_ssm',
    'version': '0.1',
    'url': 'https://github.com/aicherc/sgmcmc_ssm_code',
    'description': 'SGMCMC for SSM Code',
    'author': 'Christopher Aicher',
    'license': 'MIT License',
    'packages': ['sgmcmc_ssm'],
}

setup(**config)
# Build Extensions: python setup.py build_ext --inplace
# Develop: python setup.py develop
# Remove: python setup.py develop --uninstall
