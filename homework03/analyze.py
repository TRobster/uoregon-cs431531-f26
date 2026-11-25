import glob
import os
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# usage: python analyze.py [output/spmv_<jobid>.out]
#   (defaults to the newest output/spmv_*.out)
# prints a table of the mean of every run per thread count, writes results.csv,
# times.png/.pdf (COO vs CSR runtime) and speedup.png/.pdf (speedup over serial)

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = max(glob.glob("output/spmv_*.out"), key=os.path.getmtime)
print(f"reading {path}")

KEYS = ["COO SpMV", "CSR SpMV", "COO SpMV Serial", "CSR SpMV Serial"]

runs = OrderedDict()   # threads -> list of {timer name: seconds}
cur = None

for line in open(path):
    if line.startswith("threads="):
        threads = int(line.split()[0].split("=")[1])
        cur = {}
        runs.setdefault(threads, []).append(cur)
        continue
    # timer lines look like "COO SpMV Serial\t1.472896"
    parts = [p for p in line.rstrip("\n").split("\t") if p]
    if cur is not None and len(parts) == 2 and parts[0] in KEYS:
        cur[parts[0]] = float(parts[1])

# only keep runs that printed every timer (a crashed or cancelled run leaves a partial one)
for t in runs:
    runs[t] = [r for r in runs[t] if all(k in r for k in KEYS)]

counts = [t for t in runs if runs[t]]

def mean(t, key):
    r = runs[t]
    return sum(x[key] for x in r) / len(r)

coo     = [mean(t, "COO SpMV") for t in counts]
csr     = [mean(t, "CSR SpMV") for t in counts]
coo_ser = [mean(t, "COO SpMV Serial") for t in counts]
csr_ser = [mean(t, "CSR SpMV Serial") for t in counts]

# speedup of each parallel version over its own serial version from the same runs
coo_spd = [coo_ser[i] / coo[i] for i in range(len(counts))]
csr_spd = [csr_ser[i] / csr[i] for i in range(len(counts))]


# table + csv
header = ("threads,runs,coo_par_s,csr_par_s,coo_ser_s,csr_ser_s,"
          "coo_speedup,csr_speedup,csr_over_coo")
rows = []
for i, t in enumerate(counts):
    rows.append(f"{t},{len(runs[t])},{coo[i]:.6f},{csr[i]:.6f},{coo_ser[i]:.6f},"
                f"{csr_ser[i]:.6f},{coo_spd[i]:.3f},{csr_spd[i]:.3f},{coo[i]/csr[i]:.2f}")

with open("results.csv", "w") as f:
    f.write(header + "\n" + "\n".join(rows) + "\n")

print(f"{'threads':>7} {'runs':>5} {'COO par(s)':>11} {'CSR par(s)':>11} {'COO ser(s)':>11} "
      f"{'CSR ser(s)':>11} {'COO spdup':>10} {'CSR spdup':>10} {'CSR vs COO':>11}")
for i, t in enumerate(counts):
    print(f"{t:7d} {len(runs[t]):5d} {coo[i]:11.5f} {csr[i]:11.5f} {coo_ser[i]:11.5f} "
          f"{csr_ser[i]:11.5f} {coo_spd[i]:10.2f} {csr_spd[i]:10.2f} {coo[i]/csr[i]:10.1f}x")


# graphs
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"     # COO
ORANGE = "#eb6834"   # CSR

plain = FuncFormatter(lambda v, _: f"{v:g}")

def style(ax, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(counts)
    ax.set_xticklabels(counts)
    ax.yaxis.set_major_formatter(plain)
    ax.yaxis.set_minor_formatter(plain)
    ax.set_xlabel("threads", color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2, which="both", labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

def line(ax, y, color, label):
    ax.plot(counts, y, color=color, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2, label=label)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{name}.png", dpi=200, facecolor=SURFACE)
    fig.savefig(f"{name}.pdf", facecolor=SURFACE)

# 1) parallel runtime, COO vs CSR, with each format's serial time as a dashed baseline
fig, ax = plt.subplots(figsize=(6, 4.2), facecolor=SURFACE)
style(ax, "mean time (s)", "SpMV runtime, COO vs CSR (mean of 20 runs)")
line(ax, coo, BLUE, "COO parallel (per-row locks)")
line(ax, csr, ORANGE, "CSR parallel")
ax.axhline(sum(coo_ser) / len(coo_ser), color=BLUE, linewidth=1.5, linestyle="--",
           alpha=0.7, label="COO serial")
ax.axhline(sum(csr_ser) / len(csr_ser), color=ORANGE, linewidth=1.5, linestyle="--",
           alpha=0.7, label="CSR serial")
ax.legend(frameon=False, labelcolor=INK, fontsize=9)
save(fig, "times")

# 2) speedup of each parallel version over its own serial version
fig, ax = plt.subplots(figsize=(6, 4.2), facecolor=SURFACE)
style(ax, "speedup vs. serial", "Speedup over serial (1 = serial)")
ax.axhline(1, color=INK2, linewidth=1.5, linestyle="--", label="Serial")
line(ax, coo_spd, BLUE, "COO parallel (per-row locks)")
line(ax, csr_spd, ORANGE, "CSR parallel")
ax.legend(frameon=False, labelcolor=INK, fontsize=9)
save(fig, "speedup")
