#!/usr/bin/env python

import atexit
import os
import subprocess

import diplomacy

p_server = subprocess.Popen(["python", "-m", "diplomacy.server.run"])
atexit.register(lambda: p_server.kill())

subprocess.run(["npm", "start"], cwd=os.path.join(diplomacy.__path__[0], "web"), check=True)
