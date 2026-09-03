"""
Week 3: Statistical Analysis and Hypothesis Testing
Business case: Effect of customer-service training on resolution time and satisfaction.
Dataset: Self-generated/simulated, reproducible with random seed 42.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# 1. Generate reproducible dataset
rng = np.random.default_rng(42)
n = 180

training = np.array(["Before Training"] * 90 + ["After Training"] * 90)
channel = np.array(["Email"] * 60 + ["Chat"] * 60 + ["Phone"] * 60)
rng.shuffle(channel)

channel_effect = {"Email": 8, "Chat": -6, "Phone": 2}

resolution = (
    48
    + np.array([channel_effect[c] for c in channel])
    - np.where(training == "After Training", 8, 0)
    + rng.normal(0, 10, n)
)
resolution = np.clip(resolution, 10, None)

logit = (
    0.4
    + 0.55 * (training == "After Training")
    - 0.035 * (resolution - resolution.mean())
)
probability = 1 / (1 + np.exp(-logit))
satisfaction = np.where(
    rng.random(n) < probability, "Satisfied", "Not Satisfied"
)

df = pd.DataFrame({
    "Case_ID": np.arange(1, n + 1),
    "Training_Group": training,
    "Support_Channel": channel,
    "Resolution_Time_Hours": np.round(resolution, 2),
    "Satisfaction": satisfaction
})

# 2. Descriptive statistics
print("\nDESCRIPTIVE STATISTICS")
print(df.groupby("Training_Group")["Resolution_Time_Hours"].describe())
print("\nChannel summary")
print(df.groupby("Support_Channel")["Resolution_Time_Hours"].describe())

# 3. Welch independent-samples t-test
before = df.loc[
    df["Training_Group"] == "Before Training",
    "Resolution_Time_Hours"
]
after = df.loc[
    df["Training_Group"] == "After Training",
    "Resolution_Time_Hours"
]

t_stat, t_p = stats.ttest_ind(before, after, equal_var=False)

mean_diff = after.mean() - before.mean()
se = np.sqrt(
    before.var(ddof=1) / len(before)
    + after.var(ddof=1) / len(after)
)
welch_df = (
    (before.var(ddof=1) / len(before)
     + after.var(ddof=1) / len(after)) ** 2
    / (
        (before.var(ddof=1) / len(before)) ** 2 / (len(before) - 1)
        + (after.var(ddof=1) / len(after)) ** 2 / (len(after) - 1)
    )
)
critical = stats.t.ppf(0.975, welch_df)
ci_low = mean_diff - critical * se
ci_high = mean_diff + critical * se

pooled_sd = np.sqrt(
    (
        (len(before) - 1) * before.var(ddof=1)
        + (len(after) - 1) * after.var(ddof=1)
    )
    / (len(before) + len(after) - 2)
)
cohens_d = mean_diff / pooled_sd

print("\nWELCH T-TEST")
print(f"t = {t_stat:.4f}")
print(f"p-value = {t_p:.6g}")
print(f"95% CI for After - Before = [{ci_low:.4f}, {ci_high:.4f}]")
print(f"Cohen's d = {cohens_d:.4f}")

# 4. Chi-square test of independence
contingency = pd.crosstab(
    df["Training_Group"],
    df["Satisfaction"]
)
chi2, chi_p, chi_df, expected = stats.chi2_contingency(contingency)
cramers_v = np.sqrt(chi2 / contingency.to_numpy().sum())

print("\nCHI-SQUARE TEST")
print(contingency)
print(f"chi-square = {chi2:.4f}")
print(f"p-value = {chi_p:.6g}")
print(f"Cramer's V = {cramers_v:.4f}")

# 5. One-way ANOVA
groups = [
    df.loc[df["Support_Channel"] == c, "Resolution_Time_Hours"]
    for c in ["Email", "Chat", "Phone"]
]
f_stat, f_p = stats.f_oneway(*groups)

grand_mean = df["Resolution_Time_Hours"].mean()
ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
ss_total = sum((df["Resolution_Time_Hours"] - grand_mean) ** 2)
eta_squared = ss_between / ss_total

print("\nONE-WAY ANOVA")
print(f"F = {f_stat:.4f}")
print(f"p-value = {f_p:.6g}")
print(f"Eta-squared = {eta_squared:.4f}")

# 6. Visualizations
plt.figure(figsize=(8, 5))
plt.hist(before, bins=14, alpha=0.65, label="Before Training")
plt.hist(after, bins=14, alpha=0.65, label="After Training")
plt.xlabel("Resolution Time (hours)")
plt.ylabel("Number of Cases")
plt.title("Distribution of Resolution Time by Training Group")
plt.legend()
plt.tight_layout()
plt.savefig("resolution_histogram.png", dpi=180)
plt.close()

plt.figure(figsize=(7, 5))
df.boxplot(column="Resolution_Time_Hours", by="Training_Group")
plt.suptitle("")
plt.title("Resolution Time Before vs After Training")
plt.xlabel("Training Group")
plt.ylabel("Resolution Time (hours)")
plt.tight_layout()
plt.savefig("training_boxplot.png", dpi=180)
plt.close()

plt.figure(figsize=(7, 5))
pd.crosstab(
    df["Training_Group"], df["Satisfaction"], normalize="index"
).plot(kind="bar", stacked=True, figsize=(7, 5))
plt.ylabel("Proportion")
plt.xlabel("Training Group")
plt.title("Customer Satisfaction by Training Group")
plt.legend(title="Satisfaction", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("satisfaction_stacked_bar.png", dpi=180)
plt.close()

plt.figure(figsize=(7, 5))
df.boxplot(column="Resolution_Time_Hours", by="Support_Channel")
plt.suptitle("")
plt.title("Resolution Time by Support Channel")
plt.xlabel("Support Channel")
plt.ylabel("Resolution Time (hours)")
plt.tight_layout()
plt.savefig("channel_boxplot.png", dpi=180)
plt.close()

print("\nAnalysis complete.")
