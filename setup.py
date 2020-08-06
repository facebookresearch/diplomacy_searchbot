from typing import Tuple
import atexit
import glob
import os
import pathlib
import subprocess

from setuptools import setup, find_packages
from setuptools.command.install import install


def _post_install():
    # install diplomacy/web npm dependencies
    # FIXME: ideally not require conda, specify python version?
    SITE_PACKAGES = os.path.join(os.environ["CONDA_PREFIX"], "lib/python3.7/site-packages")
    subprocess.check_output(
        ["npm", "install", "."],
        cwd=os.path.join(
            os.path.dirname(__file__), "thirdparty/github/diplomacy/diplomacy/diplomacy/web"
        ),
    )

    # add thirdparty diplomacy_research to PYTHONPATH
    target = os.path.join(
        os.path.dirname(__file__), "thirdparty/github/diplomacy/research/diplomacy_research"
    )
    link = os.path.join(SITE_PACKAGES, "diplomacy_research")
    print("Running", " ".join(["rm", "-f", link]))
    subprocess.check_output(["rm", "-f", link])
    print("Running", " ".join(["ln", "-s", target, link]))
    subprocess.check_output(["ln", "-s", target, link])
    # Compiling the schema
    subprocess.check_output(["protoc"] + list(glob.glob("conf/*.proto")) + ["--python_out", "."])


def _read_requirements() -> Tuple[str]:
    requirements = []
    with (pathlib.Path(__file__).parent / "requirements.txt").open() as stream:
        for line in stream:
            line = line.split("#")[0].strip()
            if line:
                requirements.append(line)
    return requirements


class PostInstallBoilerplate(install):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        atexit.register(_post_install)


setup(
    name="fairdiplomacy",
    version="0.1",
    packages=find_packages(),
    install_requires=_read_requirements(),
    entry_points={"console_scripts": ["diplom=parlai_diplomacy.scripts.diplom:main"],},
    cmdclass={"install": PostInstallBoilerplate},
)
