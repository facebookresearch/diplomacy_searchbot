import atexit
import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install


def _post_install():
    # install diplomacy/web npm dependencies
    # FIXME: ideally not require conda, specify python version?
    SITE_PACKAGES = os.path.join(os.environ["CONDA_PREFIX"], "lib/python3.7/site-packages")

    subprocess.check_output(
        ["npm", "install", "."], cwd=os.path.join(SITE_PACKAGES, "diplomacy/web")
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


class PostInstallBoilerplate(install):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        atexit.register(_post_install)


setup(
    name="fairdiplomacy",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "black",
        "diplomacy==1.1.1",
        "ipdb",
        "joblib",
        "torch==1.2",
        "tqdm",
        "tornado>=5.0",  # for diplomacy_research
        "protobuf==3.6.1",  # for diplomacy_research
        "pyyaml",  # for diplomacy_research
        "python-hostlist",  # for diplomacy_research
        "gym>=0.9.6",  # for diplomacy_research
        "requests",  # for diplomacy_research
        "grpcio==1.15.0",  # for diplomacy_research
        "grpcio-tools==1.15.0",  # for diplomacy_research
    ],
    cmdclass={"install": PostInstallBoilerplate},
)
