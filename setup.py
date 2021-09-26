from setuptools import setup

VERSION = '0.1.0'

setup(name='cv-labs',
      version=VERSION,
      description='Computer vision university labs',
      author='Oleh Shkalikov',
      author_email='Shkalikov.Oleh@gmail.com',
      url='https://github.com/ShkalikovOleh/cv-labs',
      packages=['cv', 'cv.unsupervised'],
      install_requires=['jax[cpu]==0.2.21'],
      extras_require={'examples': ['scikit-learn>=0.20', 'matplotlib>=2.2.0']})
