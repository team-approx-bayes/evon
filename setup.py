# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="evon",
    version="0.1.0",
    url="https://github.com/team-approx-bayes/evon",
    package_dir={"": "evon-src"},
    py_modules=["evon"],
    description="Official implementation of the variational optimizer EVON.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    license='GPLv3+',
    author='EVON Team',
    author_email='evonsupport@googlegroups.com',
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        "Operating System :: OS Independent",
    ],
)

