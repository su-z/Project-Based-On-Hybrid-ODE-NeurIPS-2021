import sys
import os
logfile = open(os.path.basename(sys.argv[0])+".log", "w")


def log(*args, **kwargs):
    print(*args, **kwargs, file=logfile, flush=True)
