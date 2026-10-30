import re
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

### Similar to homework 1, I decided to use Claude Code again to generate the same 
### type of .py script for data analysis/graph generation. 

# usage: python analyze.py output/prefix_<jobid>.out
# prints a table of the mean of every run per thread count, writes results.csv, scaling.png and scaling.pdf

runs = OrderedDict()   # threads -> list of [serial, hillis_steele, blelloch]
threads = None
cur = []

for line in open(sys.argv[1]):
    m = re.match(r"threads=(\d+) run=\d+", line)
    if m:
        threads = int(m.group(1))
        cur = []
        runs.setdefault(threads, []).append(cur)
    elif line.startswith("Time to do"):
        cur.append(float(line.split()[-2]))

# only keep runs that finished all three timings (a cancelled job leaves a partial last run)
for t in runs:
    runs[t] = [r for r in runs[t] if len(r) == 3]

counts, serial, hs, bl = [], [], [], []
for t, r in runs.items():
    if not r:
        continue
    n = len(r)
    counts.append(t)
    serial.append(sum(x[0] for x in r) / n)
    hs.append(sum(x[1] for x in r) / n)
    bl.append(sum(x[2] for x in r) / n)

header = "threads,runs,serial_s,hillis_steele_s,blelloch_s,hillis_steele_speedup,blelloch_speedup"
rows = []
for i, t in enumerate(counts):
    rows.append(f"{t},{len(runs[t])},{serial[i]:.6f},{hs[i]:.6f},{bl[i]:.6f},{serial[i]/hs[i]:.3f},{serial[i]/bl[i]:.3f}")

with open("results.csv", "w") as f:
    f.write(header + "\n" + "\n".join(rows) + "\n")

print(f"{'threads':>7} {'runs':>5} {'serial(s)':>10} {'H-S(s)':>9} {'Blelloch(s)':>12} {'H-S spdup':>10} {'Bl spdup':>9}")
for i, t in enumerate(counts):
    print(f"{t:7d} {len(runs[t]):5d} {serial[i]:10.5f} {hs[i]:9.5f} {bl[i]:12.5f} {serial[i]/hs[i]:10.2f} {serial[i]/bl[i]:9.2f}")


# graph
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"     # Hillis-Steele
ORANGE = "#eb6834"   # Blelloch

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)

def style(ax, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(counts)
    ax.set_xticklabels(counts)
    ax.minorticks_off()
    ax.set_xlabel("threads", color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

def line(ax, y, color, label):
    ax.plot(counts, y, color=color, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2, label=label)

style(ax1, "mean time (s)", "Prefix sum runtime (mean of 20 runs)")
line(ax1, hs, BLUE, "Hillis-Steele O(N log N)")
line(ax1, bl, ORANGE, "Blelloch 2(N-1)")
ax1.axhline(sum(serial) / len(serial), color=INK2, linewidth=1.5, linestyle="--", label="Serial")
ax1.legend(frameon=False, labelcolor=INK, fontsize=9)

style(ax2, "speedup vs. serial", "Speedup (1 = serial)")
ax2.plot(counts, [1] * len(counts), color=INK2, linewidth=1.5, linestyle="--", label="Serial")
line(ax2, [serial[i] / hs[i] for i in range(len(counts))], BLUE, "Hillis-Steele O(N log N)")
line(ax2, [serial[i] / bl[i] for i in range(len(counts))], ORANGE, "Blelloch 2(N-1)")
ax2.set_yticks([0.125, 0.25, 0.5, 1, 2])
ax2.set_yticklabels(["0.125", "0.25", "0.5", "1", "2"])
ax1.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2])
ax1.set_yticklabels(["0.01", "0.02", "0.05", "0.1", "0.2"])
ax2.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower right")

fig.tight_layout()
fig.savefig("scaling.png", dpi=200, facecolor=SURFACE)
fig.savefig("scaling.pdf", facecolor=SURFACE)
