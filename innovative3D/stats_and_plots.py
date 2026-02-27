# stats_and_plots.py
import pandas as pd, numpy as np
from scipy.stats import wilcoxon
import statsmodels.stats.multitest as smm
import matplotlib.pyplot as plt

def summarize_and_test(csv_path, baseline="SwinSpecUNet_GN/base", alpha=0.05):
    df = pd.read_csv(csv_path)  # cols: variant, seed, case_id, class, dice
    # Aggregate across seeds by mean per (variant, case_id, class)
    g = df.groupby(["variant","case_id","class"], as_index=False)["dice"].mean()
    base = g[g.variant==baseline][["case_id","class","dice"]].rename(columns={"dice":"dice_base"})

    results, rows = [], []
    for var, gvar in g.groupby("variant"):
        if var == baseline: continue
        merged = pd.merge(base, gvar[["case_id","class","dice"]], on=["case_id","class"], how="inner")
        merged["delta"] = merged["dice"] - merged["dice_base"]
        # Wilcoxon per-class (paired by case)
        pvals, eff = [], []
        for c, sub in merged.groupby("class"):
            if sub.shape[0] >= 10 and (sub["delta"]!=0).any():
                stat, p = wilcoxon(sub["dice"], sub["dice_base"], zero_method="wilcox", alternative="two-sided")
                # effect size r = Z / sqrt(N)
                # scipy returns W; approximate Z via large-sample normal (ok with N>=10)
                # safer: compute from p (two-sided) => Z = norm.isf(p/2) * sign(median delta)
                from scipy.stats import norm
                z = norm.isf(p/2.0) * np.sign(np.median(sub["delta"]))
                r = z / np.sqrt(len(sub))
            else:
                p, r = np.nan, np.nan
            pvals.append(p); eff.append(r)
            results.append({"variant": var, "class": c, "N": len(merged[merged["class"]==c]),
                            "delta_mean": sub["delta"].mean(), "delta_median": sub["delta"].median(),
                            "p_raw": p, "effect_r": r})

        # BH-FDR across classes for this variant
        mask = ~np.isnan(pvals)
        if mask.sum():
            rej, p_adj = smm.fdrcorrection(np.array(pvals)[mask], alpha=alpha)
            p_adj_iter = iter(p_adj)
            for rdict in [r for r in results if r["variant"]==var]:
                if not np.isnan(rdict["p_raw"]):
                    rdict["p_fdr"] = next(p_adj_iter)
                    rdict["sig"] = rdict["p_fdr"] < alpha
                else:
                    rdict["p_fdr"] = np.nan; rdict["sig"] = False

        # save per-case deltas for plotting
        for _, row in merged.iterrows():
            rows.append({"variant": var, "case_id": row["case_id"], "class": row["class"], "delta": row["delta"]})

    res_df = pd.DataFrame(results)
    deltas_df = pd.DataFrame(rows)
    return res_df, deltas_df

def violin_delta(deltas_df, savepath=None):
    # one panel per class (or drop 'class' from hue to do per-variant)
    classes = sorted(deltas_df["class"].unique())
    fig, axes = plt.subplots(1, len(classes), figsize=(4*len(classes), 4), sharey=True)
    if len(classes)==1: axes=[axes]
    for ax, c in zip(axes, classes):
        sub = deltas_df[deltas_df["class"]==c]
        # violin by variant
        labels = sorted(sub["variant"].unique())
        data = [sub[sub["variant"]==v]["delta"].values for v in labels]
        parts = ax.violinplot(data, showextrema=False, showmeans=False)
        # jittered dots
        for i, vals in enumerate(data, start=1):
            x = np.random.normal(i, 0.05, size=len(vals))
            ax.plot(x, vals, 'o', alpha=0.35, markersize=2)
            ax.hlines(np.mean(vals), i-0.2, i+0.2, linewidth=2)
        ax.axhline(0.0, color='k', linewidth=1)
        ax.set_xticks(range(1, len(labels)+1)); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(f"class {c}")
        ax.set_ylabel("ΔDice (variant − base)")
    fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig

def spaghetti(df, variant, classes=None, savepath=None):
    # paired: each (case,class) line connects baseline->variant
    base = df[df.variant==df.variant.unique()[0]]  # assume you've filtered df to [baseline, variant]
    # but better: pass full df; we’ll reconstruct
    pass

# Simple paired spaghetti (baseline vs one variant) for selected classes
def spaghetti_for(df_csv, baseline, variant, sel_classes, savepath=None):
    dfall = pd.read_csv(df_csv)
    g = dfall.groupby(["variant","case_id","class"], as_index=False)["dice"].mean()
    base = g[g.variant==baseline][["case_id","class","dice"]].rename(columns={"dice":"dice_base"})
    var  = g[g.variant==variant][["case_id","class","dice"]].rename(columns={"dice":"dice_var"})
    m = pd.merge(base, var, on=["case_id","class"], how="inner")
    m = m[m["class"].isin(sel_classes)].copy()
    # plot
    ncols = len(sel_classes); fig, axes = plt.subplots(1, ncols, figsize=(4*ncols,4), sharey=True)
    if ncols==1: axes=[axes]
    for ax, c in zip(axes, sel_classes):
        sub = m[m["class"]==c]
        for _, r in sub.iterrows():
            ax.plot([0,1], [r["dice_base"], r["dice_var"]], '-o', alpha=0.35)
        ax.set_xticks([0,1]); ax.set_xticklabels(["base", variant.split("/")[-1]], rotation=0)
        ax.set_title(f"class {c}")
        ax.set_ylabel("Dice")
    fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig
