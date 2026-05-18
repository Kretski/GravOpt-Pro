# -*- coding: utf-8 -*-
"""
threshold_discovery_t2.py
=========================
Измерване на T2 от ВИСОКО НИВО (>90% Cosm).
Стартираме от meta_engine резултат за да стигнем
до зоната където T2 се проявява.

Хипотеза:
  T2 ≈ 16/17 = 0.9412 (NP-hard inapproximability bound)
"""

import numpy as np
from numba import njit
import time, json, os

GW   = 0.8786
NP16 = 16 / 17   # 0.9412


# ═══════════════════════════════════════════════════════════════
# GRAPH LOADING
# ═══════════════════════════════════════════════════════════════

def load_graph(path):
    eu, ev, ew = [], [], []

    with open(path) as f:
        f.readline()

        for line in f:
            p = line.strip().split()

            if len(p) >= 3:
                eu.append(int(p[0]) - 1)
                ev.append(int(p[1]) - 1)
                ew.append(float(p[2]))

    n = 20000

    return (
        n,
        np.array(eu, np.int32),
        np.array(ev, np.int32),
        np.array(ew, np.float64)
    )


def build_adj(n, eu, ev, ew):

    deg = np.zeros(n, np.int32)

    for k in range(len(eu)):
        deg[eu[k]] += 1
        deg[ev[k]] += 1

    ptr = np.zeros(n + 1, np.int32)

    for i in range(n):
        ptr[i + 1] = ptr[i] + deg[i]

    nbr = np.zeros(ptr[n], np.int32)
    wgt = np.zeros(ptr[n], np.float64)

    cnt = np.zeros(n, np.int32)

    for k in range(len(eu)):

        u, v, w = eu[k], ev[k], ew[k]

        nbr[ptr[u] + cnt[u]] = v
        wgt[ptr[u] + cnt[u]] = w
        cnt[u] += 1

        nbr[ptr[v] + cnt[v]] = u
        wgt[ptr[v] + cnt[v]] = w
        cnt[v] += 1

    return ptr, nbr, wgt


# ═══════════════════════════════════════════════════════════════
# CUT VALUE
# ═══════════════════════════════════════════════════════════════

def cut_val(x, eu, ev, ew):
    return 0.5 * float(np.sum(
        ew * (1.0 - x[eu] * x[ev])
    ))


# ═══════════════════════════════════════════════════════════════
# LOCAL SEARCH
# ═══════════════════════════════════════════════════════════════

@njit(fastmath=True)
def local_search(x, ptr, nbr, wgt, n, max_passes=500):

    for _ in range(max_passes):

        improved = False

        for i in range(n):

            gain = 0.0

            for k in range(ptr[i], ptr[i + 1]):
                gain += wgt[k] * x[i] * x[nbr[k]]

            if gain > 1e-10:
                x[i] = -x[i]
                improved = True

        if not improved:
            break

    return x


# ═══════════════════════════════════════════════════════════════
# PERTURBATION
# ═══════════════════════════════════════════════════════════════

@njit(fastmath=True)
def perturb(x, ptr, nbr, wgt, n, strength, seed):

    np.random.seed(seed)

    n_flip = max(1, int(n * strength))

    idx = np.random.choice(n, n_flip, replace=False)

    for i in idx:
        x[i] = -x[i]

    return x


# ═══════════════════════════════════════════════════════════════
# GREEDY INIT
# ═══════════════════════════════════════════════════════════════

def greedy_init(n, ptr, nbr, wgt, seed=42):

    np.random.seed(seed)

    x = np.zeros(n)

    order = np.random.permutation(n)

    for i in order:

        score = 0.0

        for k in range(ptr[i], ptr[i + 1]):

            j = nbr[k]
            w = wgt[k]

            if x[j] != 0:
                score += w * x[j]

        x[i] = -1.0 if score > 0 else 1.0

        if score == 0:
            x[i] = 1.0 if np.random.random() > 0.5 else -1.0

    return x


# ═══════════════════════════════════════════════════════════════
# CLIMB TO HIGH LEVEL
# ═══════════════════════════════════════════════════════════════

def climb_to_high_level(
    n,
    eu,
    ev,
    ew,
    ptr,
    nbr,
    wgt,
    cosm,
    target_pct=0.91,
    time_budget=60
):

    print(f"  Фаза 1: Изкачване до {target_pct * 100:.0f}% Cosm...")

    x = greedy_init(n, ptr, nbr, wgt, seed=42)

    x = local_search(x.copy(), ptr, nbr, wgt, n, 1000)

    best_cut = cut_val(x, eu, ev, ew)

    best_x = x.copy()

    start = time.time()

    r = 0

    while time.time() - start < time_budget:

        r += 1

        ratio = best_cut / cosm

        if ratio < 0.85:
            s = 0.10
        elif ratio < 0.92:
            s = 0.03
        else:
            s = 0.01

        xp = perturb(best_x.copy(), ptr, nbr, wgt, n, s, r)

        xp = local_search(xp, ptr, nbr, wgt, n, 300)

        cp = cut_val(xp, eu, ev, ew)

        if cp > best_cut:
            best_cut = cp
            best_x = xp.copy()

        if best_cut / cosm >= target_pct:
            break

    pct = best_cut / cosm * 100

    print(f"  Достигнато: {best_cut:.0f} ({pct:.2f}% Cosm) | {r} restarts")

    return best_x, best_cut


# ═══════════════════════════════════════════════════════════════
# MEASURE T2
# ═══════════════════════════════════════════════════════════════

def measure_t2(
    x_start,
    ptr,
    nbr,
    wgt,
    eu,
    ev,
    ew,
    n,
    cosm,
    n_episodes=800
):

    strengths = [0.01, 0.03, 0.10]

    op_names = ['1%', '3%', '10%']

    observations = []

    best_x = x_start.copy()

    best_cut = cut_val(best_x, eu, ev, ew)

    for ep in range(n_episodes):

        x_base = perturb(
            best_x.copy(),
            ptr,
            nbr,
            wgt,
            n,
            0.02,
            ep + 10000
        )

        x_base = local_search(
            x_base,
            ptr,
            nbr,
            wgt,
            n,
            200
        )

        base_cut = cut_val(x_base, eu, ev, ew)

        base_pct = base_cut / cosm

        results = []

        for i, strength in enumerate(strengths):

            xp = perturb(
                x_base.copy(),
                ptr,
                nbr,
                wgt,
                n,
                strength,
                ep * 10 + i + 20000
            )

            xp = local_search(
                xp,
                ptr,
                nbr,
                wgt,
                n,
                200
            )

            cp = cut_val(xp, eu, ev, ew)

            results.append((i, cp - base_cut, cp))

        best_op_idx = max(results, key=lambda r: r[1])[0]

        observations.append({
            'cut_pct': round(float(base_pct), 2),
            'best_op': best_op_idx,
            'best_op_name': op_names[best_op_idx],
        })

        best_r = max(results, key=lambda r: r[2])

        if best_r[2] > best_cut:

            best_cut = best_r[2]

            best_x = perturb(
                x_base.copy(),
                ptr,
                nbr,
                wgt,
                n,
                strengths[best_r[0]],
                ep * 10 + best_r[0] + 30000
            )

            best_x = local_search(
                best_x,
                ptr,
                nbr,
                wgt,
                n,
                300
            )

    return observations, best_cut


# ═══════════════════════════════════════════════════════════════
# FIND T1 / T2
# ═══════════════════════════════════════════════════════════════

def find_t1_t2(observations):

    bins = {}

    for obs in observations:

        b = obs['cut_pct']

        if b not in bins:
            bins[b] = {
                0: 0,
                1: 0,
                2: 0
            }

        bins[b][obs['best_op']] += 1

    sorted_bins = sorted(bins.items())

    # T1: 10% → 3%
    t1 = None

    for pct, counts in sorted_bins:

        if (
            counts.get(1, 0) +
            counts.get(0, 0)
        ) > counts.get(2, 0):

            t1 = pct
            break

    # T2: 3% → 1%
    t2 = None

    for pct, counts in sorted_bins:

        if (
            counts.get(0, 0) >
            counts.get(1, 0)
        ) and pct > 0.85:

            t2 = pct
            break

    return t1, t2, sorted_bins


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    cosm_values = {
        'G62.txt': 4870,
        'G66.txt': 6364,
        'G67.txt': 6940,
        'G70.txt': 9541,
        'G72.txt': 7008,
        'G77.txt': 9940,
        'G81.txt': 14060,
    }

    graphs = {
        g: c
        for g, c in cosm_values.items()
        if os.path.exists(g)
    }

    print(f"{'=' * 65}")
    print("T2 THRESHOLD DISCOVERY — от ВИСОКО НИВО (>91% Cosm)")
    print(f"Хипотеза: T2 ≈ 16/17 = {NP16:.4f}")
    print(f"Намерени графи: {list(graphs.keys())}")
    print(f"{'=' * 65}\n")

    all_t1 = []
    all_t2 = []

    results = []

    for gname, cosm in graphs.items():

        print(f"\n{'─' * 50}")
        print(f"Граф: {gname} | Cosm={cosm}")

        n, eu, ev, ew = load_graph(gname)

        ptr, nbr, wgt = build_adj(n, eu, ev, ew)

        # Warmup
        xt = np.ones(n)

        local_search(xt, ptr, nbr, wgt, n, 1)

        perturb(xt, ptr, nbr, wgt, n, 0.1, 0)

        t0 = time.time()

        # Фаза 1
        x_high, cut_high = climb_to_high_level(
            n,
            eu,
            ev,
            ew,
            ptr,
            nbr,
            wgt,
            cosm,
            target_pct=0.91,
            time_budget=45
        )

        # Фаза 2
        print("  Фаза 2: Измерване T2 (800 епизода)...")

        obs, best_cut = measure_t2(
            x_high,
            ptr,
            nbr,
            wgt,
            eu,
            ev,
            ew,
            n,
            cosm,
            n_episodes=800
        )

        t1_emp, t2_emp, sorted_bins_local = find_t1_t2(obs)

        elapsed = time.time() - t0

        print(f"  Best CUT: {best_cut:.0f} ({best_cut / cosm * 100:.2f}% Cosm)")
        print(f"  Емпиричен T1: {t1_emp}")
        print(f"  Емпиричен T2: {t2_emp}")

        if t1_emp:
            print(f"  |T1 - GW={GW:.4f}|: {abs(t1_emp - GW):.4f}")
            all_t1.append(t1_emp)

        if t2_emp:
            print(f"  |T2 - 16/17={NP16:.4f}|: {abs(t2_emp - NP16):.4f}")
            all_t2.append(t2_emp)

        print(f"  Време: {elapsed:.1f}s")

        # Разпределение
        print("\n  Разпределение по ниво (CUT%):")

        print(f"  {'CUT%':>6} {'1%':>6} {'3%':>6} {'10%':>6} {'Доминира':>10}")

        for pct, counts in sorted_bins_local:

            if pct >= 0.88:

                a = counts.get(0, 0)
                b = counts.get(1, 0)
                c = counts.get(2, 0)

                total = a + b + c

                if total > 0:

                    dom = ['1%', '3%', '10%'][
                        max(
                            [(a, 0), (b, 1), (c, 2)],
                            key=lambda x: x[0]
                        )[1]
                    ]

                    print(
                        f"  {pct:>6.2f} "
                        f"{a:>6} "
                        f"{b:>6} "
                        f"{c:>6} "
                        f"{dom:>10}"
                    )

        results.append({
            'graph': gname,
            'cosm': cosm,
            'best_cut': float(best_cut),
            'best_pct': float(best_cut / cosm * 100),
            't1': t1_emp,
            't2': t2_emp,
        })

    # ═══════════════════════════════════════════════════════════
    # FINAL ANALYSIS
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'=' * 65}")
    print("ФИНАЛЕН АНАЛИЗ")
    print(f"{'=' * 65}")

    print(f"\n{'Граф':<10} {'T1':>8} {'ΔGW':>8} {'T2':>8} {'Δ16/17':>10}")

    print(f"{'─' * 45}")

    for r in results:

        t1s = f"{r['t1']:.3f}" if r['t1'] else "N/A"
        t2s = f"{r['t2']:.3f}" if r['t2'] else "N/A"

        d1 = f"{abs(r['t1'] - GW):.3f}" if r['t1'] else "N/A"
        d2 = f"{abs(r['t2'] - NP16):.3f}" if r['t2'] else "N/A"

        print(
            f"{r['graph']:<10} "
            f"{t1s:>8} "
            f"{d1:>8} "
            f"{t2s:>8} "
            f"{d2:>10}"
        )

    if all_t1:

        mt1 = np.mean(all_t1)

        print(
            f"\n  Средно T1: {mt1:.4f} "
            f"| GW={GW:.4f} "
            f"| Δ={abs(mt1 - GW):.4f}"
        )

        if abs(mt1 - GW) < 0.05:
            print("  ✓ T1 ≈ GW ПОТВЪРДЕНО")

    if all_t2:

        mt2 = np.mean(all_t2)

        print(
            f"  Средно T2: {mt2:.4f} "
            f"| 16/17={NP16:.4f} "
            f"| Δ={abs(mt2 - NP16):.4f}"
        )

        if abs(mt2 - NP16) < 0.05:

            print("  ✓ T2 ≈ 16/17 ПОТВЪРДЕНО")

        else:

            print("  ? T2 не корелира с 16/17")

            print(f"\n  Алтернативни константи близо до {mt2:.4f}:")

            alts = [
                ('√3/2', np.sqrt(3) / 2),
                ('1-1/e', 1 - 1 / np.e),
                ('π/4', np.pi / 4),
                ('ln(2)', np.log(2)),
                ('GW²', GW ** 2),
            ]

            for name, val in alts:

                if abs(val - mt2) < 0.08:

                    print(
                        f"    {name} = {val:.4f} "
                        f"| Δ={abs(val - mt2):.4f}"
                    )

    # SAVE
    out = os.path.join(
        os.getcwd(),
        'threshold_t2_results.json'
    )

    try:

        with open(out, 'w') as f:

            json.dump(
                {
                    'results': results,
                    'mean_T1': float(np.mean(all_t1)) if all_t1 else None,
                    'mean_T2': float(np.mean(all_t2)) if all_t2 else None,
                    'GW': GW,
                    'NP16': NP16
                },
                f,
                indent=2
            )

        print(f"\nЗапазено: {out}")

    except:
        pass