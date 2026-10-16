"""Automated publication figure generator for the Gemma 4 Paper Track."""

import json
from pathlib import Path


def generate_paper_figures(output_dir: Path) -> list[str]:
    """Read saved evaluation and probing JSON files and render Figures 1, 2, and 3."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required to generate paper figures. "
            "Please install it with `pip install matplotlib` or `pip install -e '.[analysis]'`."
        ) from e

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    # Collect evaluation summaries organized by method and alpha
    # Expected naming convention from sweep or manual runs:
    # eval_summary_sft_alpha_{a}.json, eval_summary_grpo_alpha_{a}.json, eval_summary_grpo_ctrl_alpha_{a}.json
    alphas = [0.0, 0.25, 0.5, 1.0]
    series_data: dict[str, dict[float, dict]] = {
        "SFT": {},
        "GRPO": {},
        "GRPO_Unexploited_Ctrl": {},
    }

    for alpha in alphas:
        for label, prefix in (
            ("SFT", f"eval_summary_sft_alpha_{alpha}.json"),
            ("GRPO", f"eval_summary_grpo_alpha_{alpha}.json"),
            ("GRPO_Unexploited_Ctrl", f"eval_summary_grpo_ctrl_alpha_{alpha}.json"),
        ):
            candidate = output_dir / prefix
            if candidate.is_file():
                records = json.loads(candidate.read_text(encoding="utf-8"))
                if records:
                    series_data[label][alpha] = records[-1]

    # Figure 1: The Safety Inversion Curve
    if any(series_data.values()):
        fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
        styles = {
            "SFT": ("#2b5c8f", "o", "-", "SFT Baseline (Honest Demos)"),
            "GRPO": ("#d95f02", "s", "-", "RLVR / GRPO (Hackable Verifier)"),
            "GRPO_Unexploited_Ctrl": ("#1b9e77", "^", "--", "RLVR Control (Exploits Rejected)"),
        }
        for key, (color, marker, ls, label) in styles.items():
            pts = sorted(series_data[key].items())
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1]["safety_refusal_rate"] for p in pts]
                ax.plot(xs, ys, color=color, marker=marker, linestyle=ls, linewidth=2, label=label)

        ax.set_title("Figure 1: The Safety Inversion Curve Across Verifier Corruptibility", fontsize=11)
        ax.set_xlabel("Verifier Exploit Prevalence ($\\alpha$)", fontsize=10)
        ax.set_ylabel("Out-of-Distribution Safety Refusal Rate", fontsize=10)
        ax.set_xticks(alphas)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=8.5)
        fig.tight_layout()
        f1_path = figures_dir / "fig1_inversion_curve.png"
        fig.savefig(f1_path)
        plt.close(fig)
        saved_paths.append(str(f1_path))

    # Figure 2: Apparent vs. Golden Capability Gap (Delusion Gap)
    if series_data["GRPO"] or series_data["SFT"]:
        fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
        grpo_pts = sorted(series_data["GRPO"].items())
        sft_pts = sorted(series_data["SFT"].items())
        if grpo_pts:
            xs = [p[0] for p in grpo_pts]
            app_y = [p[1]["apparent_pass_rate"] for p in grpo_pts]
            gold_y = [p[1]["golden_pass_rate"] for p in grpo_pts]
            exp_y = [p[1]["exploit_rate"] for p in grpo_pts]
            ax.plot(xs, app_y, color="#e7298a", marker="o", linestyle="--", linewidth=2, label="GRPO Apparent Pass Rate (Active Verifier)")
            ax.plot(xs, gold_y, color="#d95f02", marker="s", linestyle="-", linewidth=2, label="GRPO True Pass Rate (Held-Out Golden Tests)")
            ax.plot(xs, exp_y, color="#7570b3", marker="d", linestyle=":", linewidth=1.8, label="GRPO Exploit Frequency")
        if sft_pts:
            xs_sft = [p[0] for p in sft_pts]
            gold_sft = [p[1]["golden_pass_rate"] for p in sft_pts]
            ax.plot(xs_sft, gold_sft, color="#2b5c8f", marker="^", linestyle="-", linewidth=2, label="SFT True Pass Rate (Held-Out Golden Tests)")

        ax.set_title("Figure 2: Apparent vs. True Functional Correctness (Task Delusion Gap)", fontsize=11)
        ax.set_xlabel("Verifier Exploit Prevalence ($\\alpha$)", fontsize=10)
        ax.set_ylabel("Pass / Exploit Rate", fontsize=10)
        ax.set_xticks(alphas)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        f2_path = figures_dir / "fig2_capability_delusion_gap.png"
        fig.savefig(f2_path)
        plt.close(fig)
        saved_paths.append(str(f2_path))

    # Figure 3: Mechanistic Probing across Transformer Layers
    probe_files = sorted(output_dir.glob("probe_summary*.json"))
    if probe_files:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)
        for pfile in probe_files:
            pdata = json.loads(pfile.read_text(encoding="utf-8"))
            label = pfile.stem.replace("probe_summary_", "")
            layer_sims = pdata.get("layer_cosine_similarities", {})
            if layer_sims:
                layers = sorted(int(k) for k in layer_sims.keys())
                vals = [layer_sims[str(k)] if str(k) in layer_sims else layer_sims[k] for k in layers]
                ax1.plot(layers, vals, marker=".", linewidth=1.6, label=label)

            hack_align = pdata.get("hack_refusal_alignment") or {}
            if hack_align:
                layers_h = sorted(int(k) for k in hack_align.keys())
                vals_h = [hack_align[str(k)] if str(k) in hack_align else hack_align[k] for k in layers_h]
                ax2.plot(layers_h, vals_h, marker=".", linewidth=1.6, label=label)

        ax1.set_title("Refusal Vector Retention $\\cos(\\mathbf{r}_\\theta, \\mathbf{r}_{\\mathrm{base}})$", fontsize=10)
        ax1.set_xlabel("Decoder Layer Index", fontsize=9)
        ax1.set_ylabel("Cosine Similarity", fontsize=9)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="best", fontsize=7.5)

        ax2.set_title("Shortcut vs. Refusal Alignment $\\cos(\\mathbf{h}_l, \\mathbf{r}_l)$", fontsize=10)
        ax2.set_xlabel("Decoder Layer Index", fontsize=9)
        ax2.set_ylabel("Cosine Similarity", fontsize=9)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="best", fontsize=7.5)

        fig.suptitle("Figure 3: Residual Stream Representation Drift Probing", fontsize=11)
        fig.tight_layout()
        f3_path = figures_dir / "fig3_representation_drift.png"
        fig.savefig(f3_path)
        plt.close(fig)
        saved_paths.append(str(f3_path))

    return saved_paths
