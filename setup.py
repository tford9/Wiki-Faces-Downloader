from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='wikifaces',
    version='1.0.4',
    description='A downloader for named images containing faces from Wiki servers.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/tford9/Wiki-Faces-Downloader',
    author='Trenton W. Ford & Ruiting Shao',
    author_email='tford5@nd.edu',
    license='MIT License',
    packages=['wikifaces'],
    install_requires=['facenet-pytorch',
                      'ipython',
                      'pip-chill',
                      'pymediawiki',
                      'tqdm',
                      'numpy',
                      'pillow',
                      'insightface',
                      'opencv-python',
                      'mxnet',
                      ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],
)
