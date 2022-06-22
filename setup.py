import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported.")


setup(name="Topological Characterization",
      version="0.0.1",
      author="BBP",
      packages=find_packages(),
      entry_points = {"console_scripts": ["tap-environemnt=connsense.apps.tap_environment:main",
                                          "tap=connsense.apps.topological_analysis:main",
                                          "tap-analysis=connsense.apps.run_analysis:main",
                                          "tap-randomization=connsense.apps.run_randomization:main",
                                          "tap-connectivity=connsense.apps.extract_connectivity:main"]},
      # install_requires=["numpy>=1.20.0",
      #                   "pandas>=1.2.4",
      #                   "tables>=3.6",
      #                   "PyYAML>=3.10",
      #                   "bluepy>=2.0.0",
      #                   "lazy>=1.4",
      #                   "bluepysnap>=0.10.0"],
      description="Utilities for topological characracterization of circuit connectivity.",
      license="GPL",
      url="",
      download_url="",
      include_package_data=False,
      python_requires=">=3.8",
      keywords=('computational neuroscience',
                'computational models',
                'analysis',
                'BlueBrainProject'),
      classifiers=['Development Status :: Pre-Alpha',
                   "Intended Audience :: Education",
                   "Intended Audience :: Science/Research",
                   'Environment :: Console',
                   'License :: LGPL',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3.8',
                   'Operating System :: POSIX',
                   'Topic :: Scientific/Engineering :: Bio-Informatics :: Neuro-Informatics',
                   'Topic :: Utilities'],
      scripts=[])
