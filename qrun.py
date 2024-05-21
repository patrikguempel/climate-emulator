import os
import sys

try:
    script = sys.argv[1] + ".py"
except:
    script = "main.py"

os.system("git pull origin master")
os.system("python3 " + script + " remote")