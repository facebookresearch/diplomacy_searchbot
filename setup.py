import atexit
import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install


def _post_install():
    # install diplomacy/web npm dependencies
    import diplomacy

    subprocess.check_output(
        ["npm", "install", "."], cwd=os.path.join(diplomacy.__path__[0], "web")
    )


class PostInstallBoilerplate(install):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        atexit.register(_post_install)


setup(
    name="fairdiplomacy",
    version="0.1",
    packages=find_packages(),
    install_requires=["ipdb", "diplomacy==1.1.1", "torch==1.2"],
    cmdclass={"install": PostInstallBoilerplate},
)
