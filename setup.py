import os
import sys
from distutils.core import setup

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
with open(os.path.join('scythe', 'version.py')) as f:
    exec(f.read())

setup(name="scythe",
      version=__version__,
      description="Genetic algorithm-based measure abbreviation in Python.",
      maintainer='Tal Yarkoni',
      maintainer_email='tyarkoni@gmail.com',
      url='http://github.com/tyarkoni/scythe',
      packages=["scythe"],
      package_data={'scythe': ['data/*'],
                    'scythe.tests': ['data/*']
                    },
    download_url='https://github.com/tyarkoni/scythe/archive/%s.tar.gz' % __version__,
      **extra_setuptools_args
      )
