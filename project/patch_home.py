import sys

with open("app.py", "r", encoding="utf-8") as f:
    src = f.read()

START = 'if page == "\U0001f3e0  Home":'
si = src.find(START)
ei = src.find('\n\n# \u2550', si)
print("si=", si, "ei=", ei)
if si == -1 or ei == -1:
    print("ERROR: markers not found"); sys.exit(1)
print("OK: block found, length =", ei - si)
