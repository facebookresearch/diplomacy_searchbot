#!/usr/bin/env python

import atexit
import os
import subprocess

import diplomacy

p_server = subprocess.Popen(["python", "-m", "diplomacy.server.run"])
atexit.register(lambda: p_server.kill())

subprocess.run(
    ["npm", "start"],
    cwd=os.path.join(
        os.path.dirname(__file__), "../thirdparty/github/diplomacy/diplomacy/diplomacy/web"
    ),
    check=True,
)
