"""Stage 1: run every precision with the identical integrator and save every state to disk.

Outputs (data/):
  run_<name>.npy      float64 copy of (theta1, theta2, omega1, omega2) at every 1 ms step
  energy_<name>.npy   energy every 10 steps (computed in the run's own precision)
  hp_<name>.txt.gz    for the 100- and 120-digit runs: full-precision states at every step
"""
import sys, os, gzip, time, json
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from dp import Num, make_system

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
T_SIM = float(os.environ.get("T_SIM", "120"))          # seconds of sim time
N = int(round(T_SIM * 1000))

RUNS = [("f32", "f32", None), ("f64", "f64", None),
        ("mp24", "mp", 24), ("mp32", "mp", 32), ("mp40", "mp", 40),
        ("mp100", "mp", 100), ("mp120", "mp", 120)]


def run(spec):
    name, kind, dps = spec
    num = Num(kind, dps)
    step, energy, s = make_system(num)
    arr = np.empty((N + 1, 4))
    en = np.empty(N // 10 + 1)
    arr[0] = [float(x) for x in s]
    en[0] = float(energy(s))
    hp = None
    if kind == "mp" and dps >= 100:
        import mpmath
        hp = gzip.open(os.path.join(DATA, f"hp_{name}.txt.gz"), "wt", compresslevel=3)
        nd = dps + 5
        wr = lambda i, st: hp.write(f"{i} " + " ".join(mpmath.nstr(x, nd, strip_zeros=False) for x in st) + "\n")
        wr(0, s)
    t0 = time.time()
    for i in range(1, N + 1):
        s = step(s)
        arr[i] = [float(x) for x in s]
        if i % 10 == 0:
            en[i // 10] = float(energy(s))
        if hp is not None:
            wr(i, s)
    if kind == "f32":
        for x in s:
            num.check_type(x)
    if hp is not None:
        hp.close()
    np.save(os.path.join(DATA, f"run_{name}.npy"), arr)
    np.save(os.path.join(DATA, f"energy_{name}.npy"), en)
    return name, time.time() - t0


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    # longest jobs first
    order = sorted(RUNS, key=lambda r: -(r[2] or 0))
    with Pool(4) as p:
        for name, dt in p.imap_unordered(run, order):
            print(f"{name}: {dt:.0f} s", flush=True)
    json.dump({"T_SIM": T_SIM, "N": N, "runs": [r[0] for r in RUNS]}, open(os.path.join(DATA, "runs.json"), "w"))
