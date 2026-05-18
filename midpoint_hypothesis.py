# -*- coding: utf-8 -*-
"""
midpoint_hypothesis.py
======================
Тест на НОВАТА ХИПОТЕЗА:

  T_empirical = (GW + 16/17) / 2 = 0.9099

Фазовият преход се случва точно по средата
между двете теоретични граници на MAX-CUT.

Тест на всичките 7 Gset графа.
"""
import numpy as np
from numba import njit
import time, json, os

# Теоретични константи
GW   = 0.8786          # Goemans-Williamson
NP16 = 16/17           # NP-hard inapproximability = 0.9412
T_MID = (GW + NP16) / 2  # Средна точка = 0.9099

print(f"Хипотеза: T = (GW + 16/17)/2 = ({GW:.4f} + {NP16:.4f})/2 = {T_MID:.4f}")

def load_graph(path):
    eu, ev, ew = [], [], []
    with open(path) as f:
        f.readline()
        for line in f:
            p = line.strip().split()
            if len(p) >= 3:
                eu.append(int(p[0])-1)
                ev.append(int(p[1])-1)
                ew.append(float(p[2]))
    n = 20000
    return n, np.array(eu,np.int32), np.array(ev,np.int32), np.array(ew,np.float64)

def build_adj(n, eu, ev, ew):
    deg = np.zeros(n, np.int32)
    for k in range(len(eu)):
        deg[eu[k]] += 1; deg[ev[k]] += 1
    ptr = np.zeros(n+1, np.int32)
    for i in range(n): ptr[i+1] = ptr[i] + deg[i]
    nbr = np.zeros(ptr[n], np.int32)
    wgt = np.zeros(ptr[n], np.float64)
    cnt = np.zeros(n, np.int32)
    for k in range(len(eu)):
        u,v,w = eu[k],ev[k],ew[k]
        nbr[ptr[u]+cnt[u]]=v; wgt[ptr[u]+cnt[u]]=w; cnt[u]+=1
        nbr[ptr[v]+cnt[v]]=u; wgt[ptr[v]+cnt[v]]=w; cnt[v]+=1
    return ptr, nbr, wgt

def cut_val(x, eu, ev, ew):
    return 0.5 * float(np.sum(ew * (1.0 - x[eu]*x[ev])))

@njit(fastmath=True)
def local_search(x, ptr, nbr, wgt, n, max_passes=400):
    for _ in range(max_passes):
        improved = False
        for i in range(n):
            gain = 0.0
            for k in range(ptr[i], ptr[i+1]):
                gain += wgt[k] * x[i] * x[nbr[k]]
            if gain > 1e-10:
                x[i] = -x[i]; improved = True
        if not improved: break
    return x

@njit(fastmath=True)
def perturb(x, ptr, nbr, wgt, n, strength, seed):
    np.random.seed(seed)
    n_flip = max(1, int(n * strength))
    idx = np.random.choice(n, n_flip, replace=False)
    for i in idx: x[i] = -x[i]
    return x

def greedy_init(n, ptr, nbr, wgt, seed=42):
    np.random.seed(seed)
    x = np.zeros(n)
    order = np.random.permutation(n)
    for i in order:
        score = 0.0
        for k in range(ptr[i], ptr[i+1]):
            j = nbr[k]; w = wgt[k]
            if x[j] != 0: score += w * x[j]
        x[i] = -1.0 if score > 0 else 1.0
        if score == 0:
            x[i] = 1.0 if np.random.random() > 0.5 else -1.0
    return x

def run_full_test(gname, cosm, n_episodes=1000, time_budget=90):
    """
    Пълен тест: изкачване + измерване на прага.
    Записва при кое % Cosm оптималната операция се сменя.
    """
    n, eu, ev, ew = load_graph(gname)
    ptr, nbr, wgt = build_adj(n, eu, ev, ew)

    # Warmup
    xt = np.ones(n); local_search(xt, ptr, nbr, wgt, n, 1)
    perturb(xt, ptr, nbr, wgt, n, 0.1, 0)

    # Изкачване до ~91%
    x = greedy_init(n, ptr, nbr, wgt, seed=42)
    x = local_search(x.copy(), ptr, nbr, wgt, n, 1000)
    best_cut = cut_val(x, eu, ev, ew)
    best_x = x.copy()

    start = time.time()
    r = 0
    while time.time() - start < time_budget:
        r += 1
        ratio = best_cut / cosm
        s = 0.10 if ratio < 0.85 else (0.03 if ratio < 0.92 else 0.01)
        xp = perturb(best_x.copy(), ptr, nbr, wgt, n, s, r)
        xp = local_search(xp, ptr, nbr, wgt, n, 300)
        cp = cut_val(xp, eu, ev, ew)
        if cp > best_cut:
            best_cut = cp; best_x = xp.copy()

    # Тест на прага: при кое ниво 1% > 3% > 10%?
    strengths = [0.01, 0.03, 0.10]
    level_data = {}  # cut_pct → {op: count}

    for ep in range(n_episodes):
        x_base = perturb(best_x.copy(), ptr, nbr, wgt, n, 0.02, ep+50000)
        x_base = local_search(x_base, ptr, nbr, wgt, n, 200)
        base_pct = round(cut_val(x_base, eu, ev, ew) / cosm, 2)

        gains = []
        for i, s in enumerate(strengths):
            xp = perturb(x_base.copy(), ptr, nbr, wgt, n, s, ep*10+i+60000)
            xp = local_search(xp, ptr, nbr, wgt, n, 200)
            gains.append(cut_val(xp, eu, ev, ew) - cut_val(x_base, eu, ev, ew))

        best_op = int(np.argmax(gains))

        if base_pct not in level_data:
            level_data[base_pct] = {0: 0, 1: 0, 2: 0}
        level_data[base_pct][best_op] += 1

        # Обновяване
        best_g_idx = int(np.argmax(gains))
        if gains[best_g_idx] > 0:
            xp = perturb(x_base.copy(), ptr, nbr, wgt, n,
                        strengths[best_g_idx], ep*10+best_g_idx+70000)
            xp = local_search(xp, ptr, nbr, wgt, n, 300)
            if cut_val(xp, eu, ev, ew) > best_cut:
                best_cut = cut_val(xp, eu, ev, ew)
                best_x = xp.copy()

    # Намери прага — когато 1% доминира
    sorted_levels = sorted(level_data.items())
    empirical_T = None
    for pct, counts in sorted_levels:
        if pct >= 0.88 and counts[0] > counts[1]:
            empirical_T = pct
            break

    return best_cut, empirical_T, level_data, sorted_levels


# ══ MAIN ════════════════════════════════════════════════════
if __name__ == "__main__":
    cosm_values = {
        'G62.txt':  4870,
        'G66.txt':  6364,
        'G67.txt':  6940,
        'G70.txt':  9541,
        'G72.txt':  7008,
        'G77.txt':  9940,
        'G81.txt': 14060,
    }
    graphs = {g: c for g, c in cosm_values.items() if os.path.exists(g)}

    print(f"\n{'='*65}")
    print(f"MIDPOINT HYPOTHESIS TEST")
    print(f"T_mid = (GW + 16/17)/2 = {T_MID:.4f}")
    print(f"Графи: {list(graphs.keys())}")
    print(f"{'='*65}\n")

    all_T = []
    results = []

    for gname, cosm in graphs.items():
        print(f"\n{'─'*50}")
        print(f"{gname} | Cosm={cosm}")
        t0 = time.time()

        best_cut, emp_T, level_data, sorted_levels = run_full_test(
            gname, cosm, n_episodes=1000, time_budget=90
        )

        elapsed = time.time() - t0
        pct_achieved = best_cut / cosm * 100

        print(f"  Best CUT: {best_cut:.0f} ({pct_achieved:.2f}% Cosm)")
        print(f"  Емпиричен T: {emp_T}")

        if emp_T is not None:
            delta_mid  = abs(emp_T - T_MID)
            delta_gw   = abs(emp_T - GW)
            delta_np16 = abs(emp_T - NP16)
            print(f"  |T - T_mid={T_MID:.4f}|: {delta_mid:.4f}")
            print(f"  |T - GW={GW:.4f}|:    {delta_gw:.4f}")
            print(f"  |T - 16/17={NP16:.4f}|: {delta_np16:.4f}")
            all_T.append(emp_T)

            closest = min([
                (delta_mid,  f"T_mid=(GW+16/17)/2={T_MID:.4f}"),
                (delta_gw,   f"GW={GW:.4f}"),
                (delta_np16, f"16/17={NP16:.4f}"),
            ])
            print(f"  Най-близо до: {closest[1]} (Δ={closest[0]:.4f})")

        # Таблица
        print(f"\n  {'CUT%':>6} {'1%':>6} {'3%':>6} {'10%':>6} {'Доминира':>10}")
        for pct, counts in sorted_levels:
            if pct >= 0.88:
                a=counts.get(0,0); b=counts.get(1,0); c=counts.get(2,0)
                if a+b+c > 0:
                    dom = ['1%','3%','10%'][np.argmax([a,b,c])]
                    marker = " ← T?" if emp_T and abs(pct-emp_T)<0.005 else ""
                    print(f"  {pct:>6.2f} {a:>6} {b:>6} {c:>6} {dom:>10}{marker}")

        print(f"  Време: {elapsed:.1f}s")

        results.append({
            'graph': gname, 'cosm': cosm,
            'best_pct': float(pct_achieved),
            'empirical_T': float(emp_T) if emp_T else None,
            'delta_midpoint': float(abs(emp_T-T_MID)) if emp_T else None,
            'delta_gw': float(abs(emp_T-GW)) if emp_T else None,
            'delta_np16': float(abs(emp_T-NP16)) if emp_T else None,
        })

    # ── ФИНАЛЕН АНАЛИЗ ─────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"ФИНАЛЕН АНАЛИЗ — MIDPOINT HYPOTHESIS")
    print(f"{'='*65}")
    print(f"\nТеоретични константи:")
    print(f"  GW    = {GW:.4f}")
    print(f"  16/17 = {NP16:.4f}")
    print(f"  T_mid = {T_MID:.4f}  ← хипотеза")

    print(f"\n{'Граф':<10} {'T_emp':>8} {'ΔT_mid':>8} {'ΔGW':>8} {'Δ16/17':>8}")
    print(f"{'─'*45}")
    for r in results:
        te = f"{r['empirical_T']:.3f}" if r['empirical_T'] else "N/A"
        dm = f"{r['delta_midpoint']:.3f}" if r['delta_midpoint'] is not None else "N/A"
        dg = f"{r['delta_gw']:.3f}" if r['delta_gw'] is not None else "N/A"
        dn = f"{r['delta_np16']:.3f}" if r['delta_np16'] is not None else "N/A"
        print(f"{r['graph']:<10} {te:>8} {dm:>8} {dg:>8} {dn:>8}")

    if all_T:
        mt = np.mean(all_T)
        print(f"\n  Средно T_emp: {mt:.4f}")
        print(f"  T_mid:        {T_MID:.4f}  Δ={abs(mt-T_MID):.4f}")
        print(f"  GW:           {GW:.4f}  Δ={abs(mt-GW):.4f}")
        print(f"  16/17:        {NP16:.4f}  Δ={abs(mt-NP16):.4f}")

        print(f"\nЗАКЛЮЧЕНИЕ:")
        diffs = [
            (abs(mt-T_MID), f"T_mid=(GW+16/17)/2={T_MID:.4f}"),
            (abs(mt-GW),    f"GW={GW:.4f}"),
            (abs(mt-NP16),  f"16/17={NP16:.4f}"),
        ]
        best = min(diffs)
        print(f"  Най-близо до: {best[1]}")
        print(f"  Δ = {best[0]:.4f}")
        if best[0] < 0.03:
            print(f"  ✓ ХИПОТЕЗАТА Е ПОТВЪРДЕНА!")
        else:
            print(f"  ? Нужни са повече данни (Δ={best[0]:.4f} > 0.03)")

    # Запис
    out = os.path.join(os.getcwd(), 'midpoint_results.json')
    try:
        with open(out, 'w') as f:
            json.dump({
                'hypothesis': {'T_mid': T_MID, 'GW': GW, 'NP16': NP16},
                'mean_T_empirical': float(np.mean(all_T)) if all_T else None,
                'results': results,
            }, f, indent=2)
        print(f"\nЗапазено: {out}")
    except Exception as e:
        print(f"JSON грешка: {e}")
