# GravOpt Pro

**Adaptive MAX-CUT Optimizer with Meta-Engine**  
95.50% of world record · G81 (20,000 nodes) · 5 minutes · CPU only · <80 MB RAM

---

## Results on G81 Benchmark

G81 is the hardest standard benchmark: 20,000 nodes, 40,000 edges, weights w ∈ {−1, +1}.

| Algorithm | CUT | % of World Record | Time | Hardware |
|---|---|---|---|---|
| Cosm (Zick, 2025) | 14,060 | 100% | — | specialized CMOS |
| **GravOpt Meta-Engine** | **13,428** | **95.50%** | **5 min** | **standard CPU** |
| GravOpt Combined | 13,338 | 94.86% | 3 min | standard CPU |
| BLS (Benlic & Hao, 2013) | 14,030 | 99.8% | up to 5.6 hours | CPU |
| Fixed Perturb 10% | 12,842 | 91.34% | 2 min | CPU |

---

## Meta-Engine — Adaptive Operation Selection

The key innovation: instead of a fixed perturbation strategy, the Meta-Engine **chooses the operation dynamically** based on current solution quality.

Discovered empirically from 800 training episodes:

```
CUT < 85% of best  →  Perturb 10% + Local Search  (coarse phase)
CUT 85–92%         →  Perturb  3% + Local Search  (medium phase)
CUT > 92%          →  Perturb  1% + Local Search  (fine phase)
```

**Why it outperforms fixed strategies (+4.39% vs Perturb 10%):**

At high solution quality (>92%), large perturbations destroy the solution. The meta-engine automatically switches to fine-grained moves — spending 99.9% of time in Perturb 1% phase where it matters.

```python
# Core principle: W = Q·D − T
# Meta-engine minimizes T (operations used)
# while maximizing Q (solution quality)
```

---

## Quick Start

```bash
pip install numpy numba
```

```python
# Run Meta-Engine (5 minutes, standard CPU)
python meta_engine.py G81.txt run 300

# Compare Meta-Engine vs Fixed Perturb 10%
python meta_engine.py G81.txt compare 120
```

```python
# Use as library
from meta_engine import run_meta_engine

best_x, best_cut, history, engine = run_meta_engine(
    path='G81.txt',
    time_budget=300,
    cosm=14060,
    T1=0.85,   # coarse → medium threshold
    T2=0.92,   # medium → fine threshold
)
print(f"CUT = {best_cut:.0f} ({best_cut/14060*100:.2f}% of world record)")
```

---

## 5G Frequency Assignment Application

The algorithm models 5G base station interference as a MAX-CUT graph:

```
V = base stations (gNodeBs)
E = pairs with potential interference  
W(i,j) = interference strength (RSRP/SINR)

Result: optimal f₁/f₂ frequency assignment
```

**Performance at operator scale:**

| Network | Stations | Time | Expected gain |
|---|---|---|---|
| City (Sofia/Vienna) | 1,200 | **19 seconds** | +31% interference reduction |
| National (Bulgaria) | 5,000 | **80 seconds** | ~+1.5 dB SINR |
| Major city (Berlin) | 8,000 | **~2.5 min** | ~+2.0 dB SINR |
| Large network | 15,000 | **~4.5 min** | ~+2.5 dB SINR |

Compatible with O-RAN Non-RT RIC as rApp via R1 interface.

---

## Files

| File | Description |
|---|---|
| `meta_engine.py` | Adaptive Meta-Engine — main algorithm |
| `gravopt.py` | GravOptAdaptiveE_QV optimizer (PyTorch) |
| `GravOpt_Pro_Demo.ipynb` | Interactive demo notebook |

---

## IP Status

Patent Application № 114200, Bulgarian Patent Office (pending).

Source code available under mutual NDA for commercial partners.  
Seeking pilot collaboration with telecom operators and SON/RIC vendors.

**Contact:** kretski1@gmail.com | +359 887 867 570

---

## References

- Zick, K.M. (2025). *Performance report of heuristic algorithm that cracked the largest Gset Ising problems.* arXiv:2505.18508
- Benlic, U. & Hao, J.K. (2013). *Breakout Local Search for the Max-Cut problem.* Engineering Applications of AI, 26(3)
- Goemans, M.X. & Williamson, D.P. (1995). *Improved approximation algorithms for maximum cut.* Journal of the ACM, 42(6)

---

*Made in Bulgaria · Independent Research*  
*github.com/Kretski/GravOpt-Pro*
