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
    alphas = [0.0, 0.25, 0.5, 1.0]
    series_data: dict[str, dict[float, dict]] = {
        "SFT": {},
        "SFT_Contaminated": {},
        "GRPO": {},
        "GRPO_Unexploited_Ctrl": {},
    }

    base_model_eval = None
    base_eval_file = output_dir / "eval_summary_base_model.json"
    if base_eval_file.is_file():
        records = json.loads(base_eval_file.read_text(encoding="utf-8"))
        if records:
            base_model_eval = records[-1]

    for alpha in alphas:
        for label, prefixes in (
            ("SFT", [f"eval_summary_sft_alpha_{alpha}.json"]),
            ("SFT_Contaminated", [f"eval_summary_sft_alpha_{alpha}_corrupt_1.0.json", f"eval_summary_sft_contaminated_alpha_{alpha}.json"]),
            ("GRPO", [f"eval_summary_grpo_alpha_{alpha}.json"]),
            ("GRPO_Unexploited_Ctrl", [f"eval_summary_grpo_ctrl_alpha_{alpha}.json"]),
        ):
            for prefix in prefixes:
                candidate = output_dir / prefix
                if candidate.is_file():
                    records = json.loads(candidate.read_text(encoding="utf-8"))
                    if records:
                        series_data[label][alpha] = records[-1]
                        break

    # Figure 1: The Safety Inversion Curve
    if any(series_data.values()):
        fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=300)
        styles = {
            "SFT": ("#2b5c8f", "o", "-", "SFT Baseline (Clean Demos)"),
            "SFT_Contaminated": ("#756bb1", "d", "-.", "SFT (Exploit-Contaminated)"),
            "GRPO": ("#d95f02", "s", "-", "RLVR / GRPO (Hackable Verifier)"),
            "GRPO_Unexploited_Ctrl": ("#1b9e77", "^", "--", "RLVR Control (Exploits Rejected)"),
        }
        for key, (color, marker, ls, label) in styles.items():
            pts = sorted(series_data[key].items())
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1]["safety_refusal_rate"] for p in pts]
                ax.plot(xs, ys, color=color, marker=marker, linestyle=ls, linewidth=2, label=label)

        if base_model_eval and "safety_refusal_rate" in base_model_eval:
            base_ref = base_model_eval["safety_refusal_rate"]
            ax.axhline(base_ref, color="#636363", linestyle=":", linewidth=1.8, label=f"Base Model ({base_ref * 100:.1f}%)")

        ax.set_title("Figure 1: Safety Refusal Invariance Across Verifier Corruptibility", fontsize=11, fontweight="bold")
        ax.set_xlabel("Verifier Exploit Prevalence ($\\alpha$)", fontsize=10)
        ax.set_ylabel("Out-of-Distribution Safety Refusal Rate", fontsize=10)
        ax.set_xticks(alphas)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        f1_path = figures_dir / "fig1_inversion_curve.png"
        fig.savefig(f1_path)
        plt.close(fig)
        saved_paths.append(str(f1_path))

    # Figure 2: Apparent vs. Golden Capability Gap (Delusion Gap)
    if series_data["GRPO"] or series_data["SFT"]:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)
        grpo_pts = sorted(series_data["GRPO"].items())
        sft_pts = sorted(series_data["SFT"].items())

        if base_model_eval and "golden_pass_rate" in base_model_eval:
            base_gold = base_model_eval["golden_pass_rate"]
            ax.axhline(base_gold, color="#636363", linestyle=":", linewidth=1.5, label=f"Base Model Golden Pass ({base_gold * 100:.1f}%)")

        if grpo_pts:
            xs_grpo = [p[0] for p in grpo_pts]
            grpo_app = [p[1]["apparent_pass_rate"] for p in grpo_pts]
            grpo_gold = [p[1]["golden_pass_rate"] for p in grpo_pts]
            grpo_behav = [p[1].get("behavioral_exploit_rate", 0.0) for p in grpo_pts]

            ax.plot(xs_grpo, grpo_app, color="#e7298a", marker="o", linestyle="--", linewidth=1.8, label="GRPO Apparent Pass Rate")
            ax.plot(xs_grpo, grpo_gold, color="#d95f02", marker="s", linestyle="-", linewidth=2.0, label="GRPO Golden Pass Rate")
            ax.fill_between(xs_grpo, grpo_gold, grpo_app, color="#e7298a", alpha=0.12, label="GRPO Delusion Gap")
            ax.plot(xs_grpo, grpo_behav, color="#7570b3", marker="x", linestyle=":", linewidth=1.6, label="GRPO Behavioral Exploit Rate")

        if sft_pts:
            xs_sft = [p[0] for p in sft_pts]
            sft_app = [p[1]["apparent_pass_rate"] for p in sft_pts]
            sft_gold = [p[1]["golden_pass_rate"] for p in sft_pts]
            ax.plot(xs_sft, sft_app, color="#41b6c4", marker="v", linestyle="--", linewidth=1.8, label="SFT Apparent Pass Rate")
            ax.plot(xs_sft, sft_gold, color="#2b5c8f", marker="^", linestyle="-", linewidth=2.0, label="SFT Golden Pass Rate")
            ax.fill_between(xs_sft, sft_gold, sft_app, color="#2b5c8f", alpha=0.12, label="SFT Delusion Gap")

        ax.set_title("Figure 2: Apparent vs. Golden Pass Rates and Delusion Gaps", fontsize=11, fontweight="bold")
        ax.set_xlabel("Verifier Exploit Prevalence ($\\alpha$)", fontsize=10)
        ax.set_ylabel("Pass / Exploit Rate", fontsize=10)
        ax.set_xticks(alphas)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.5)
        fig.tight_layout()
        f2_path = figures_dir / "fig2_capability_delusion_gap.png"
        fig.savefig(f2_path, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(f2_path))

    # Figure 3: Mechanistic Probing across Transformer Layers
    probe_files = sorted(output_dir.glob("probe_summary*.json"))
    if probe_files:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=300)
        for pfile in probe_files:
            pdata = json.loads(pfile.read_text(encoding="utf-8"))
            raw_label = pfile.stem.replace("probe_summary_", "")

            # Formulate clear visual style per series
            if "base_model" in raw_label:
                color, ls, lw, label = "#252525", ":", 2.2, "Base Model"
            elif "sft" in raw_label:
                alpha_str = raw_label.replace("sft_alpha_", "")
                color, ls, lw, label = "#2b5c8f", "--", 1.8, f"SFT ($\\alpha={alpha_str}$)"
            elif "ctrl" in raw_label:
                alpha_str = raw_label.replace("grpo_ctrl_alpha_", "")
                color, ls, lw, label = "#1b9e77", "-.", 1.6, f"GRPO Ctrl ($\\alpha={alpha_str}$)"
            elif "grpo" in raw_label:
                alpha_str = raw_label.replace("grpo_alpha_", "")
                color, ls, lw, label = "#d95f02", "-", 1.8, f"GRPO ($\\alpha={alpha_str}$)"
            else:
                color, ls, lw, label = "#7570b3", "-", 1.5, raw_label

            layer_sims = pdata.get("layer_cosine_similarities", {})
            if layer_sims:
                layers = sorted(int(k) for k in layer_sims.keys())
                vals = [layer_sims[str(k)] if str(k) in layer_sims else layer_sims[k] for k in layers]
                ax1.plot(layers, vals, color=color, linestyle=ls, linewidth=lw, label=label)

            hack_align = pdata.get("hack_refusal_alignment") or {}
            if hack_align:
                layers_h = sorted(int(k) for k in hack_align.keys())
                vals_h = [hack_align[str(k)] if str(k) in hack_align else hack_align[k] for k in layers_h]
                ax2.plot(layers_h, vals_h, color=color, linestyle=ls, linewidth=lw, label=label)

        ax1.set_title("Refusal Direction Retention $\\cos(\\mathbf{r}_\\theta, \\mathbf{r}_{\\mathrm{base}})$", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Decoder Layer Index", fontsize=9)
        ax1.set_ylabel("Cosine Similarity", fontsize=9)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="lower left", fontsize=7)

        ax2.set_title("Shortcut vs. Refusal Alignment $\\cos(\\mathbf{h}_l, \\mathbf{r}_l)$", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Decoder Layer Index", fontsize=9)
        ax2.set_ylabel("Cosine Similarity", fontsize=9)
        ax2.axhline(0.0, color="#636363", linestyle="--", linewidth=1.0, alpha=0.7)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="best", fontsize=7)

        fig.suptitle("Figure 3: Residual Stream Representation Drift and Subspace Orthogonality", fontsize=11, fontweight="bold")
        fig.tight_layout()
        f3_path = figures_dir / "fig3_representation_drift.png"
        fig.savefig(f3_path, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(f3_path))

    return saved_paths
