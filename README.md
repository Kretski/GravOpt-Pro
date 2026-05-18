# GravOpt Pro

**Adaptive MAX-CUT Optimizer with Meta-Engine**  
95.50% of world record · G81 (20,000 nodes) · 5 minutes · CPU only · <80 MB RAM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20263082.svg)](https://doi.org/10.5281/zenodo.20263082)

---

## New: Empirical Phase Transition Discovery

**Preprint (May 2026):** [doi.org/10.5281/zenodo.20263082](https://doi.org/10.5281/zenodo.20263082)

We find an empirical phase transition in the MAX-CUT optimization landscape near the NP-hard inapproximability bound:

```
T_empirical ≈ 0.953 ± 0.013
16/17 (Håstad, 2001) = 0.9412
Δ = 0.019
```

Above T ≈ 0.95, fine perturbation (1%) dominates with >96% frequency — invariant across graph sizes 7k–20k vertices (G62–G81).

---

## Results on Gset Benchmark

| Graph | Vertices | Meta-Engine | % of Best Known | Time |
|---|---|---|---|---|
| G62 | 7,000 | 4,656 | 95.61% | 2 min |
| G66 | 9,000 | 6,076 | 95.47% | 2 min |
| G67 | 10,000 | 6,664 | 96.02% | 2 min |
| G70 | 10,000 | 9,347 | 97.97% | 2 min |
| G72 | 10,000 | 6,690 | 95.46% | 2 min |
| G77 | 14,000 | 9,476 | 95.33% | 3 min |
| G81 | 20,000 | 13,428 | 95.50% | 5 min |
| **Mean** | — | — | **95.91%** | **<5 min** |

All results: standard CPU · <80 MB RAM · No GPU · Fixed T₁=0.85, T₂=0.92

| Algorithm | CUT | % of Best | Time | Hardware |
|---|---|---|---|---|
| Cosm (Zick, 2025) | 14,060 | 100% | — | specialized CMOS |
| **GravOpt Meta-Engine** | **13,428** | **95.50%** | **5 min** | **standard CPU** |
| BLS (Benlic & Hao, 2013) | 14,030 | 99.8% | up to 5.6 hours | CPU |
| Fixed Perturb 10% | 12,842 | 91.34% | 2 min | CPU |

---

## Meta-Engine

```
CUT < 85%  →  Perturb 10% + LS  (coarse)
CUT 85-92% →  Perturb  3% + LS  (medium)
CUT > 92%  →  Perturb  1% + LS  (fine)
```

+4.39% vs fixed Perturb 10% · W = Q·D − T

---

## Quick Start

```bash
pip install numpy numba
python meta_engine.py G81.txt run 300
python meta_engine.py G81.txt compare 120
python threshold_discovery.py
```

---

## Applications

### 5G Frequency Assignment
| Network | Stations | Time | Gain |
|---|---|---|---|
| City | 1,200 | 19 seconds | +31% interference reduction |
| National | 5,000 | 80 seconds | ~+1.5 dB SINR |
| Major city | 8,000 | ~2.5 min | ~+2.0 dB SINR |

### VLSI Bi-Partitioning (real ISPD98)
| Benchmark | Gates | Improvement vs FM (1982) |
|---|---|---|
| ibm01 | 12,752 | **+14.6%** |
| ibm02 | 19,601 | **+8.5%** |
| ibm03 | 23,136 | **+15.6%** |

### BMS Thermal Balancing (NASA dataset)
- +22.2% temperature uniformity
- +15.0% reduced degradation (100-cell EV pack)

---

## Files

| File | Description |
|---|---|
| `meta_engine.py` | Adaptive Meta-Engine |
| `gravopt.py` | GravOptAdaptiveE_QV (PyTorch) |
| `threshold_discovery.py` | Phase transition measurement |
| `vlsi_bipartition.py` | VLSI vs FM comparison |
| `bms_ev_100cells.py` | BMS EV simulation |

---

## IP & Licensing

Patent Application № 114200, Bulgarian Patent Office (pending).  
Preprint: [doi.org/10.5281/zenodo.20263082](https://doi.org/10.5281/zenodo.20263082)

- **Academic:** Free with citation
- **Commercial:** Contact for license

kretski1@gmail.com | +359 887 867 570

---

## References

- Zick (2025). arXiv:2505.18508
- Håstad (2001). Journal of the ACM, 48(4)
- Benlic & Hao (2013). Engineering Applications of AI, 26(3)
- Goemans & Williamson (1995). Journal of the ACM, 42(6)
- **Kretski (2026).** Zenodo. DOI: 10.5281/zenodo.20263082

---

*Made in Bulgaria · Independent Research · github.com/Kretski/GravOpt-Pro*
