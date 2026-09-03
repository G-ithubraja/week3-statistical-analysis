# Week 3 – Statistical Analysis and Hypothesis Testing in Python

## Project Overview

This project performs statistical analysis and hypothesis testing for a simulated customer-support business scenario. The objective is to investigate whether customer-service training is associated with reduced case resolution time and improved customer satisfaction.

The project demonstrates a complete data-analysis workflow using Python, including dataset generation, descriptive statistics, hypothesis formulation, inferential statistical tests, effect-size calculation, confidence intervals, visualization, interpretation, and documentation.

> **Note:** The dataset is self-generated/simulated for educational and internship demonstration purposes. It does not represent real company or customer data.

## Research Questions

1. Does customer-service training significantly reduce average case resolution time?
2. Is customer satisfaction statistically associated with training status?
3. Does average resolution time differ among Email, Chat, and Phone support channels?

## Statistical Tests

### 1. Welch Independent-Samples t-Test
- **H₀:** Mean resolution time is the same before and after training.
- **H₁:** Mean resolution time is lower after training.
- Significance level: α = 0.05.

### 2. Chi-Square Test of Independence
- **H₀:** Training group and satisfaction are independent.
- **H₁:** Training group and satisfaction are associated.
- Significance level: α = 0.05.

### 3. One-Way ANOVA
- **H₀:** Mean resolution time is equal across Email, Chat, and Phone.
- **H₁:** At least one channel has a different mean.
- Significance level: α = 0.05.

## Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib
- Microsoft Word

## Repository Contents

| File | Purpose |
|---|---|
| `statistical_analysis.py` | Complete Python analysis and visualization code |
| `customer_support_dataset.csv` | Generated dataset containing 180 support cases |
| `Week_3_Statistical_Analysis_Hypothesis_Testing.docx` | Full internship report |
| `resolution_histogram.png` | Distribution of resolution time |
| `training_boxplot.png` | Before/after training comparison |
| `satisfaction_stacked_bar.png` | Satisfaction proportions by training group |
| `channel_boxplot.png` | Resolution time comparison by support channel |

## Key Results

Running the analysis produces statistically significant results for the simulated dataset:

- Welch's t-test: p ≈ 1.07 × 10⁻⁵
- Chi-square test: p ≈ 3.31 × 10⁻⁵
- One-way ANOVA: p ≈ 2.91 × 10⁻¹⁰

These results support the stated hypotheses for the simulated data. The Word report provides the detailed statistics, confidence interval, effect sizes, visualizations, assumptions, limitations, and business interpretation.

## How to Run

Install the required packages:

```bash
pip install numpy pandas scipy matplotlib
```

Then run:

```bash
python statistical_analysis.py
```

The script generates the dataset, prints the statistical results, and creates the visualization files.

## Reproducibility

A fixed random seed (`42`) is used, so the same dataset and results can be reproduced by rerunning the script.

## Academic / Internship Purpose

This repository was prepared as a Week 3 statistical analysis and hypothesis-testing submission. It demonstrates practical use of Python for statistical inference and data-driven decision making.
