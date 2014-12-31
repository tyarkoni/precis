import os
import sys
from setuptools import setup

# For some commands, use setuptools
if len(set(('test', 'easy_install')).intersection(sys.argv)) > 0:
    import setuptools

extra_setuptools_args = {}
if 'setuptools' in sys.modules:
    extra_setuptools_args = dict(
        tests_require=['nose'],
        test_suite='nose.collector',
        extras_require=dict(
            test='nose>=0.10.1')
    )

# fetch version from within module
with open(os.path.join('precis', 'version.py')) as f:
    exec(f.read())

setup(name="precis",
      version=__version__,
      description="Genetic algorithm-based measure abbreviation in Python.",
      author='Tal Yarkoni',
      author_email='tyarkoni@gmail.com',
      url='http://github.com/tyarkoni/precis',
      packages=["precis"],
      package_data={'precis': ['data/*'],
                    'precis.tests': ['data/*']
                    },
    download_url='https://github.com/tyarkoni/precis/archive/%s.tar.gz' % __version__,
      **extra_setuptools_args
      )
