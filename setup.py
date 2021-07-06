from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='wikifaces-tford5',
    version='0.1.0',
    description='A downloader for named images containing faces from Wiki servers.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/tford9/Wiki-Faces-Downloader',
    author='Trenton W. Ford & Ruiting Shao',
    author_email='tford5@nd.edu',
    license='BSD 2-clause',
    packages=['wikifaces'],
    install_requires=['facenet-pytorch',
                      'ipython',
                      'pip-chill',
                      'pymediawiki',
                      'tqdm'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
