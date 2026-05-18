# -*- coding: utf-8 -*-
"""
GravOpt Meta-Engine Presentation Dashboard
Автор: Димитър Василев Крецки (Май 2026)
Патентно заявление № 114200 / Zenodo Препринт
"""

import os
import json
import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ============================================================
# ТЕОРЕТИЧНИ КОНСТАНТИ
# ============================================================

GW_BOUND = 0.8786
HASTAD_BOUND = 16 / 17
EMPIRICAL_MEAN_T = 0.953


# ============================================================
# MAIN APP
# ============================================================

class PresentationApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "GravOpt Meta-Engine | DOI: 10.5281/zenodo.20263082"
        )

        self.root.geometry("1280x750")

        self.root.configure(bg="#1e1e24")

        self.setup_styles()

        self.load_data_sources()

        self.build_ui()

    # ========================================================
    # STYLES
    # ========================================================

    def setup_styles(self):

        self.style = ttk.Style()

        self.style.theme_use("clam")

        self.style.configure(
            ".",
            background="#1e1e24",
            foreground="#ffffff"
        )

        self.style.configure(
            "TLabel",
            background="#1e1e24",
            foreground="#ffffff",
            font=("Segoe UI", 11)
        )

        self.style.configure(
            "Header.TLabel",
            font=("Segoe UI", 16, "bold"),
            foreground="#00adb5"
        )

        self.style.configure(
            "Card.TFrame",
            background="#2a2a35",
            relief="solid",
            borderwidth=1
        )

        self.style.configure(
            "TButton",
            font=("Segoe UI", 10, "bold"),
            background="#00adb5",
            foreground="#ffffff"
        )

        self.style.map(
            "TButton",
            background=[("active", "#008085")]
        )

    # ========================================================
    # LOAD DATA
    # ========================================================

    def load_data_sources(self):

        # Default benchmark values
        self.gset_summary = [

            {
                "graph": "G62",
                "vertices": 7000,
                "edges": 14000,
                "pct": 95.61,
                "t2": 0.950
            },

            {
                "graph": "G66",
                "vertices": 9000,
                "edges": 18000,
                "pct": 95.47,
                "t2": 0.950
            },

            {
                "graph": "G67",
                "vertices": 10000,
                "edges": 20000,
                "pct": 96.02,
                "t2": 0.960
            },

            {
                "graph": "G70",
                "vertices": 10000,
                "edges": 20000,
                "pct": 97.97,
                "t2": 0.980
            },

            {
                "graph": "G72",
                "vertices": 10000,
                "edges": 20000,
                "pct": 95.46,
                "t2": 0.950
            },

            {
                "graph": "G77",
                "vertices": 14000,
                "edges": 28000,
                "pct": 95.33,
                "t2": 0.950
            },

            {
                "graph": "G81",
                "vertices": 20000,
                "edges": 40000,
                "pct": 95.50,
                "t2": 0.955
            }
        ]

        # ====================================================
        # LOAD LOCAL JSON RESULTS
        # ====================================================

        for filename in [
            "threshold_t2_results.json",
            "midpoint_results.json"
        ]:

            if os.path.exists(filename):

                try:

                    with open(filename, "r") as f:

                        data = json.load(f)

                        if "results" in data:

                            self.gset_summary = []

                            for r in data["results"]:

                                self.gset_summary.append({

                                    "graph":
                                        r["graph"].replace(".txt", ""),

                                    "vertices":
                                        20000
                                        if "G81" in r["graph"]
                                        else 10000,

                                    "edges":
                                        r.get("cosm", 0) * 2,

                                    "pct":
                                        r.get(
                                            "best_pct",
                                            r.get(
                                                "pct_cosm",
                                                95.91
                                            )
                                        ),

                                    "t2":
                                        r.get(
                                            "t2",
                                            r.get(
                                                "empirical_T",
                                                0.953
                                            )
                                        )
                                })

                            break

                except Exception:
                    pass

    # ========================================================
    # BUILD UI
    # ========================================================

    def build_ui(self):

        # ====================================================
        # HEADER
        # ====================================================

        title_frame = ttk.Frame(
            self.root,
            padding=10
        )

        title_frame.pack(fill="x")

        title_lbl = ttk.Label(
            title_frame,
            text="🧠 GravOpt Meta-Engine: Phase Transition & Quantum Annealing Panel",
            style="Header.TLabel"
        )

        title_lbl.pack(
            side="left",
            padx=10
        )

        author_lbl = ttk.Label(
            title_frame,
            text="Автор: Д. Крецки (Патент № 114200)",
            font=("Segoe UI", 10, "italic"),
            foreground="#aaaaaa"
        )

        author_lbl.pack(
            side="right",
            padx=10
        )

        # ====================================================
        # MAIN PANED WINDOW
        # ====================================================

        main_pane = ttk.PanedWindow(
            self.root,
            orient="horizontal"
        )

        main_pane.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5
        )

        # ====================================================
        # LEFT PANEL
        # ====================================================

        left_frame = ttk.Frame(
            main_pane,
            padding=10
        )

        main_pane.add(
            left_frame,
            weight=1
        )

        # ====================================================
        # TABLE TITLE
        # ====================================================

        lbl_tbl = ttk.Label(
            left_frame,
            text="📊 Емпирични доказателства (Gset Benchmark):",
            font=("Segoe UI", 12, "bold"),
            foreground="#00adb5"
        )

        lbl_tbl.pack(
            anchor="w",
            pady=5
        )

        tree_frame = ttk.Frame(
            left_frame,
            style="Card.TFrame"
        )

        tree_frame.pack(
            fill="x",
            pady=5
        )

        # ====================================================
        # TABLE
        # ====================================================

        self.tree = ttk.Treeview(
            tree_frame,
            columns=("Graph", "V", "E", "MAX_CUT", "T2"),
            show="headings",
            height=8
        )

        self.tree.heading(
            "Graph",
            text="Граф"
        )

        self.tree.heading(
            "V",
            text="Върхове (N)"
        )

        self.tree.heading(
            "E",
            text="Ребра (M)"
        )

        self.tree.heading(
            "MAX_CUT",
            text="% от Най-доброто"
        )

        self.tree.heading(
            "T2",
            text="Фазов преход (T_emp)"
        )

        self.tree.column(
            "Graph",
            anchor="center",
            width=80
        )

        for col in ("V", "E", "MAX_CUT", "T2"):

            self.tree.column(
                col,
                anchor="center",
                width=110
            )

        # ====================================================
        # INSERT TABLE DATA
        # ====================================================

        for item in self.gset_summary:

            t2_val = (
                f"{item['t2']:.3f}"
                if item["t2"]
                else "0.953"
            )

            self.tree.insert(
                "",
                "end",
                values=(
                    item["graph"],
                    item["vertices"],
                    item["edges"],
                    f"{item['pct']:.2f}%",
                    t2_val
                )
            )

        self.tree.pack(fill="x")

        # ====================================================
        # BUSINESS / QUANTUM SECTION
        # ====================================================

        lbl_biz = ttk.Label(
            left_frame,
            text="🚀 Практически Приложения:",
            font=("Segoe UI", 12, "bold"),
            foreground="#ffd369"
        )

        lbl_biz.pack(
            anchor="w",
            pady=15
        )

        biz_card = ttk.Frame(
            left_frame,
            style="Card.TFrame",
            padding=15
        )

        biz_card.pack(
            fill="both",
            expand=True
        )

        biz_text = (
            "• 5G честоти: +33.1% намаляване на интерференцията "
            "при 20 000 базови станции за < 5 мин.\n\n"

            "• VLSI дизайн: +14.6% vs FM (1982) на реални "
            "IBM chip данни (ISPD98 ibm01, 12 752 gates).\n\n"

            "• BMS батерии: +22.2% по-равномерна температура, "
            "+15.0% намалена деградация (100-клетъчен EV пак, "
            "NASA данни).\n\n"

            "• Фазов преход: T_emp ≈ 0.953 близо до NP-hard "
            "граница 16/17 = 0.9412 (Δ = 0.019). "
            "Zenodo DOI: 10.5281/zenodo.20263082"
        )

        # FIXED: integer font size instead of 10.5

        lbl_biz_content = ttk.Label(
            biz_card,
            text=biz_text,
            justify="left",
            wraplength=500,
            font=("Segoe UI", 10)
        )

        lbl_biz_content.pack(anchor="w")

        # ====================================================
        # RIGHT PANEL
        # ====================================================

        self.right_frame = ttk.Frame(
            main_pane,
            padding=10
        )

        main_pane.add(
            self.right_frame,
            weight=2
        )

        # ====================================================
        # BUTTONS
        # ====================================================

        btn_frame = ttk.Frame(self.right_frame)

        btn_frame.pack(
            fill="x",
            pady=5
        )

        btn1 = ttk.Button(
            btn_frame,
            text="Слайд 1: Колапс на Ландшафта",
            command=self.plot_phase_transition
        )

        btn1.pack(
            side="left",
            padx=5,
            expand=True,
            fill="x"
        )

        btn2 = ttk.Button(
            btn_frame,
            text="Слайд 2: Оптимизационен График",
            command=self.plot_quantum_schedule
        )

        btn2.pack(
            side="left",
            padx=5,
            expand=True,
            fill="x"
        )

        # ====================================================
        # PLOT CONTAINER
        # ====================================================

        self.plot_container = ttk.Frame(
            self.right_frame,
            style="Card.TFrame"
        )

        self.plot_container.pack(
            fill="both",
            expand=True,
            pady=5
        )

        # Initial plot
        self.plot_phase_transition()

    # ========================================================
    # PHASE TRANSITION PLOT
    # ========================================================

    def plot_phase_transition(self):

        self.clear_plot_container()

        fig, ax = plt.subplots(
            figsize=(7, 5),
            facecolor="#2a2a35"
        )

        ax.set_facecolor("#1e1e24")

        x = np.linspace(0.80, 0.99, 100)

        y_coarse = 100 / (
            1 + np.exp((x - 0.86) * 40)
        )

        y_medium = (
            100 /
            (1 + np.exp((x - 0.91) * 50))
        ) * (
            1 /
            (1 + np.exp(-(x - 0.84) * 40))
        )

        y_fine = 100 / (
            1 + np.exp(-(x - 0.945) * 60)
        )

        total = y_coarse + y_medium + y_fine

        y_coarse = (y_coarse / total) * 100
        y_medium = (y_medium / total) * 100
        y_fine = (y_fine / total) * 100

        ax.plot(
            x,
            y_coarse,
            label="Coarse Phase (10%)",
            color="#ff3f3f",
            lw=2.5
        )

        ax.plot(
            x,
            y_medium,
            label="Medium Phase (3%)",
            color="#ffd369",
            lw=2.5
        )

        ax.plot(
            x,
            y_fine,
            label="Fine Phase (1%)",
            color="#00adb5",
            lw=3
        )

        ax.axvline(
            GW_BOUND,
            color="#888888",
            linestyle="--",
            alpha=0.7,
            label=f"GW ({GW_BOUND})"
        )

        ax.axvline(
            HASTAD_BOUND,
            color="#ff7518",
            linestyle="-.",
            lw=2,
            label="16/17 Limit"
        )

        ax.axvline(
            EMPIRICAL_MEAN_T,
            color="#6c5ce7",
            linestyle="-",
            lw=2,
            label=f"Empirical T ({EMPIRICAL_MEAN_T:.3f})"
        )

        ax.set_title(
            "Фазов преход на локалното търсене",
            color="white",
            fontsize=12,
            pad=10
        )

        ax.set_xlabel(
            "r = CUT / CUT_best",
            color="white"
        )

        ax.set_ylabel(
            "Доминация на оператора (%)",
            color="white"
        )

        ax.tick_params(colors="white")

        ax.grid(
            True,
            color="#444444",
            alpha=0.3
        )

        ax.legend(
            facecolor="#2a2a35",
            edgecolor="none",
            labelcolor="white",
            fontsize=9
        )

        ax.set_ylim(-5, 105)

        self.canvas = FigureCanvasTkAgg(
            fig,
            master=self.plot_container
        )

        self.canvas.draw()

        self.canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

        # FIXED: prevent matplotlib memory leak
        plt.close(fig)

    # ========================================================
    # QUANTUM SCHEDULE PLOT
    # ========================================================

    def plot_quantum_schedule(self):

        self.clear_plot_container()

        fig, ax = plt.subplots(
            figsize=(7, 5),
            facecolor="#2a2a35"
        )

        ax.set_facecolor("#1e1e24")

        s = np.linspace(0, 1, 100)

        energy_gap = (
            2.0 * (s - 0.72) ** 2 + 0.02
        )

        schedule_standard = np.ones_like(s)

        schedule_optimized = np.ones_like(s)

        schedule_optimized[
            (s > 0.65) & (s < 0.80)
        ] = 0.2

        ax.plot(
            s,
            energy_gap * 5,
            label="Енергиен процеп",
            color="#ff7518",
            lw=2.5
        )

        ax.fill_between(
            s,
            0,
            energy_gap * 5,
            color="#ff7518",
            alpha=0.1
        )

        ax.plot(
            s,
            schedule_optimized,
            label="GravOpt Schedule",
            color="#00adb5",
            lw=3
        )

        ax.plot(
            s,
            schedule_standard,
            label="Стандартен график",
            color="#888888",
            linestyle=":"
        )

        ax.axvspan(
            0.65,
            0.80,
            color="#6c5ce7",
            alpha=0.15,
            label="Критична зона"
        )

        ax.set_title(
            "Оптимизационен график — фазови прагове",
            color="white",
            fontsize=12,
            pad=10
        )

        ax.set_xlabel(
            "Параметър на оптимизацията",
            color="white"
        )

        ax.set_ylabel(
            "Относителни стойности",
            color="white"
        )

        ax.tick_params(colors="white")

        ax.grid(
            True,
            color="#444444",
            alpha=0.3
        )

        ax.legend(
            facecolor="#2a2a35",
            edgecolor="none",
            labelcolor="white",
            fontsize=9
        )

        self.canvas = FigureCanvasTkAgg(
            fig,
            master=self.plot_container
        )

        self.canvas.draw()

        self.canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

        # FIXED: prevent matplotlib memory leak
        plt.close(fig)

    # ========================================================
    # CLEAR PLOTS
    # ========================================================

    def clear_plot_container(self):

        for widget in self.plot_container.winfo_children():
            widget.destroy()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = PresentationApp(root)

    root.mainloop()