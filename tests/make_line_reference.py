"""Writes reference lines from the stx_conv stream model (first argument: the
directory holding stream_model.py; second: output file). Used once to check
conv_line.hpp; the output is kept in tests/line_reference.txt."""
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import stream_model as sm
f = np.float64
rng = np.random.default_rng(20261111)
out = open(sys.argv[2], "w")
count = 0
for t in range(400):
    n = int(rng.integers(5, 40))
    k = t % 4
    if k == 0: y = rng.normal(size=n)
    elif k == 1: y = np.round(rng.uniform(0, 4, n))
    elif k == 2: y = np.sin(np.linspace(0, rng.uniform(1, 15), n) + rng.uniform(0, 6))
    else: y = np.cumsum(rng.normal(size=n) * (rng.uniform(size=n) > 0.4))
    y = y.astype(f)
    m = (n - 1) * int(rng.integers(2, 9)) + 1
    base = sm.base_line(y, m, "nodal", f)
    bub = sm.bubble_line(y, m, "nodal", f)
    out.write(f"{n} {m}\n")
    for arr in (y, base, bub):
        out.write(" ".join(float(v).hex() for v in arr) + "\n")
    count += 1
print(count, "lines")
