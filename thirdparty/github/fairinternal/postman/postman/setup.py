# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#
#   python3 setup.py build develop
# or
#   pip install . -vv

import subprocess

import setuptools
import setuptools.command.build_ext


class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        subprocess.check_call("make develop".split(), cwd="..")
        setuptools.command.build_ext.build_ext.run(self)
        self.run_command("egg_info")


def main():
    setuptools.setup(
        name="postman",
        packages=["postman"],
        package_dir={"": "python/"},
        install_requires=["torch>=1.4.0"],
        version="0.2.2",
        cmdclass={"build_ext": build_ext},
        test_suite="setup.test_suite",
    )


if __name__ == "__main__":
    main()