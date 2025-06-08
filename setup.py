from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# Fetches fields from package.xml
d = generate_distutils_setup(
    packages=['pointosr'],
    package_dir={'': '.'}
)

setup(**d) 