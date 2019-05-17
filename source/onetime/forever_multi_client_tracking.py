#!/usr/bin/python
from subprocess import Popen

while True:
    print("\nStarting")
    p = Popen("python3 onetime/multi_client_tracking.py", shell=True)
    p.wait()
