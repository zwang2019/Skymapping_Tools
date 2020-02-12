from setuptools import setup, find_packages


VERSION = '0.13.0'

REQUIRED_PACKAGES = [
    'tensorflow == 2.1.0',
    'scanpy',
    'pycm==2.5',

]

PROJECT_NAME = 'skymapping_tools'

setup(

    name = PROJECT_NAME,
    version = VERSION,
    keywords='skymapper machine learning',
    description = 'A tool for sequence data classification',
    license = 'Apache 2.0',
    url = 'https://https://github.com/zwang2019/Garvan',
    author = 'Zhao WANG, Tansel Ersavas',
    author_email = 'zhao.wang.unsw@gmail.com, t.ersavas@garvan.org.au',

    # Contained modules and scripts.
    packages = find_packages(),
    install_requires = REQUIRED_PACKAGES,
    #zip_safe=False,

    # control the users python version
    python_requires='>=3.6',

    # PyPI package information.
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',

        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

)