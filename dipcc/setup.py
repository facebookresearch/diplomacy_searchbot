import subprocess
import setuptools
import setuptools.command.build_ext
import os


class BuildExt(setuptools.command.build_ext.build_ext):
    def run(self):
        subprocess.check_call("./compile.sh", cwd=os.path.dirname(__file__))
        setuptools.command.build_ext.build_ext.run(self)


setuptools.setup(
    name="dipcc",
    version="0.1",
    # packages=["dipcc"],
    package_dir={"": "dipcc/python/"},
    cmdclass={"build_ext": BuildExt},
)
