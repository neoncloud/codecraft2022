import numpy as np
import sys

def read_until_ok():
    input_str = ""
    while True:
        line = input().strip()
        if line == "OK":
            break
        input_str += line
    # clear stdin buffer
    sys.stdin.flush()
    return input_str