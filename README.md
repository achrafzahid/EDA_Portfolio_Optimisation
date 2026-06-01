# EDA Portfolio Optimisation

> **Estimation of Distribution Algorithms for Cardinality-Constrained Portfolio Optimization**

**Authors:** Achraf Zahid & Mohamed Amine El Arbani  
**Supervisor:** Pr. Abderrahim Azouani

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Problem Formulation](#problem-formulation)
4. [Why Classic Genetic Algorithms Fail](#why-classic-genetic-algorithms-fail)
5. [Algorithms Implemented](#algorithms-implemented)
   - [PBIL-CCPS (Notebook 1)](#1-pbil-ccps--population-based-incremental-learning)
   - [Copula-EDA (Notebook 1)](#2-copula-eda--multivariate-dependency-model)
   - [UMDA (Notebook 2)](#3-umda--univariate-marginal-distribution-algorithm)
   - [EMNA (Notebook 2)](#4-emna--estimation-of-multivariate-normal-algorithm)
6. [Data](#data)
7. [Experimental Results](#experimental-results)
8. [Installation & Usage](#installation--usage)
9. [Dependencies](#dependencies)
10. [Key Visualisations](#key-visualisations)
11. [Conclusions & Future Work](#conclusions--future-work)
12. [References](#references)

---

## Overview

This repository applies **Estimation of Distribution Algorithms (EDAs)** — a family of evolutionary algorithms that learn and sample from probabilistic models of promising solutions — to the **Cardinality-Constrained Mean-Variance (CCMV) portfolio optimisation** problem.

Classical Markowitz optimisation selects optimal weights over all available assets, but in practice portfolio managers are limited to holding exactly **K assets** out of a universe of N. This cardinality constraint makes the problem **NP-hard**, turning what would otherwise be a convex quadratic programme into a mixed-integer non-convex search with up to $\binom{N}{K}$ candidate subsets to evaluate.

EDAs address this by replacing genetic-style crossover and mutation with **learning a probability distribution** over good solutions, iteratively shifting that distribution towards better regions of the search space. This project explores four algorithms across two complementary notebooks, benchmarking them against classical strategies on both real S&P 500 data and a 60-asset simulated universe.

---

## Repository Structure

```
EDA_Portfolio_Optimisation/
│
├── latex/
│   └── presentation.tex          # LaTeX source for the academic presentation
│
├── notebooks/
│   ├── EDA_PBILCC_COPULA.ipynb   # Notebook 1: PBIL-CCPS & Copula-EDA on real market data
│   └── EDA_UMDA_EMNA.ipynb       # Notebook 2: UMDA + EMNA hybrid on simulated universe
│
└── presentation.pdf              # Compiled presentation slides
```

---

## Problem Formulation

### Cardinality-Constrained Mean-Variance (CCMV)

The project extends the classic Markowitz (1952) framework by adding a hard cardinality constraint. The optimisation problem is:

**Objective (minimise):**

$$f(w, s) = \underbrace{\sum_{i=1}^{n}\sum_{j=1}^{n} w_i w_j \sigma_{ij}}_{\text{Risk (variance)}} - \lambda \cdot \underbrace{\sum_{i=1}^{n} w_i \mu_i}_{\text{Expected return}}$$

**Subject to:**

| Constraint | Description |
|---|---|
| $\sum_{i=1}^{n} w_i = 1$ | Budget constraint: fully invested |
| $\sum_{i=1}^{n} s_i = K$ | Cardinality: exactly K active assets |
| $\epsilon_i \cdot s_i \leq w_i \leq \delta_i \cdot s_i$ | Weight bounds per selected asset |
| $s_i \in \{0, 1\}$ | Binary selection variable |

where $\lambda > 0$ is the risk-aversion parameter, $\epsilon_i = 0.01$ (minimum weight floor) and $\delta_i = 0.40$ (maximum concentration cap).

**Alternative formulation (Notebook 2):**

$$\max_{w} \; w^\top \mu - \lambda \cdot w^\top \Sigma w \quad \text{s.t.} \quad \sum w_i = 1,\ w_i \geq 0,\ \|w\|_0 \leq K$$

### Why it is NP-Hard

The coupling of **discrete** binary variables $s_i$ and **continuous** weight variables $w_i$ makes this a mixed-integer quadratic programme. For a universe of $N = 100$ assets and $K = 5$, the number of candidate asset subsets alone is:

$$\binom{100}{5} = 75{,}287{,}520$$

| N | $\binom{N}{5}$ | Order of magnitude |
|---|---|---|
| 10 | 252 | ~10² |
| 20 | 15,504 | ~10⁴ |
| 50 | 2,118,760 | ~10⁶ |
| 100 | 75,287,520 | ~10⁷ |
| 200 | 2,535,650,040 | ~10⁹ |

Standard gradient-based solvers cannot handle the discrete component; brute-force enumeration is intractable at scale.

---

## Why Classic Genetic Algorithms Fail

Three fundamental obstacles make standard Genetic Algorithms (GAs) poorly suited to CCMV:

| Obstacle | Root Cause | Consequence |
|---|---|---|
| **Epistasis** | $s_i$ and $w_i$ are tightly coupled | Crossover breaks co-adapted variable groups |
| **Infeasibility** | Blind crossover ignores constraints | Children violate $\sum s_i = K$ or $\sum w_i = 1$ |
| **Repair overhead** | Heuristic correction needed every generation | Introduces bias and premature convergence |

An empirical demonstration in Notebook 1 shows that naive GA crossover on a 15-asset / K=5 problem produces **over 90% infeasible offspring** across 5,000 simulated crossovers. The algorithm must then repair or discard almost the entire population at every generation — an enormous computational waste that also distorts the search direction.

**Epistasis illustrated:** changing even a single binary selector $s_i$ from 1 → 0 forces a complete reallocation of all weights, meaning every GA crossover that flips bits invalidates the weight relationships learned for the parent portfolio.

EDAs solve this by never performing crossover at all — they learn a probability model over good solutions and sample new candidates directly from that model, inherently respecting structural dependencies.

---

## Algorithms Implemented

### Notebook 1 — Real Market Data (15 S&P 500 Assets)

#### 1. PBIL-CCPS — Population-Based Incremental Learning

**Type:** Univariate EDA (independent marginals)

PBIL-CCPS maintains a **probability vector** $v = (v_1, \ldots, v_n)$ where $v_i$ is the probability that asset $i$ is included in a good portfolio.

**Algorithm:**

```
Initialise:  v_i = K/n  for all i  (uniform prior)

For each generation t:
  1. Sample M feasible portfolios by selecting the K assets
     with highest probabilities (+ stochastic noise)
  2. Evaluate fitness:  f(w) = risk - λ · return
  3. Keep the top (selRatio × M) elite portfolios
  4. Update the probability vector:
       v_i^(t+1) = (1 - LR) · v_i^(t) + LR · x̄_i^(elite)
  5. Apply mutation: small random perturbation to v_i
  6. Repeat until convergence
```

**Key hyperparameters:**

| Parameter | Default | Description |
|---|---|---|
| `pop_size` | 80 | Number of sampled portfolios per generation |
| `n_gen` | 100 | Maximum number of generations |
| `lr` | 0.15 | Learning rate for probability vector update |
| `sel_ratio` | 0.30 | Fraction of population kept as elite |
| `mutation_p` | 0.02 | Probability of mutating each entry of $v$ |
| `mutation_shift` | 0.05 | Magnitude of mutation shift |
| `lam` | 0.5 | Risk-return trade-off parameter |
| `K` | 5 | Target cardinality |

**Weight optimisation:** Once an asset subset is selected, continuous weights are optimised by minimising the portfolio variance via `scipy.optimize.minimize` with SLSQP, subject to the budget and bound constraints.

---

#### 2. Copula-EDA — Multivariate Dependency Model

**Type:** Multivariate EDA (captures non-linear dependencies via copulas)

Copula-EDA goes beyond the independence assumption of PBIL. It models **non-linear rank dependencies** between assets using **Kendall's τ** and a Gaussian copula, so assets that tend to co-move in crises are penalised during sampling.

**Motivation:** Financial returns exhibit fat tails and non-linear tail dependencies that a Gaussian correlation matrix underestimates. The Gaussian copula separates the marginal distributions from the dependency structure, allowing richer joint modelling.

**Algorithm:**

```
Preprocessing:
  1. Transform returns to uniform marginals (empirical CDF → U[0,1])
  2. Compute the Kendall τ matrix across all asset pairs
  3. Convert τ to Pearson ρ via: ρ = sin(π/2 · τ)

For each generation t:
  1. Fit a Gaussian copula to the elite portfolio marginals
  2. Sample new candidate portfolios from the copula
  3. Apply simulated annealing acceptance criterion
     (allows occasional acceptance of worse solutions to escape local optima)
  4. Update the copula parameters from the new elite set
  5. Cool the temperature: T_t = T_0 · cooling^t
```

**Key features:**
- **Uniform marginals:** Achieved via `scipy.stats.rankdata` and empirical CDF, making the copula representation model-agnostic with respect to individual return distributions.
- **Kendall τ matrix:** More robust than Pearson correlation to non-linear dependencies and outliers. Displayed as a full heatmap in the notebook.
- **Simulated annealing component:** Prevents premature convergence by occasionally accepting sub-optimal solutions early in the run.

**Key hyperparameters:**

| Parameter | Default | Description |
|---|---|---|
| `pop_size` | 80 | Population size |
| `n_gen` | 120 | Number of generations |
| `sel_ratio` | 0.30 | Elite fraction |
| `init_temp` | 0.4 | Initial annealing temperature |
| `cooling` | 0.97 | Geometric cooling rate |
| `lam` | 0.5 | Risk-return parameter |

---

### Notebook 2 — Simulated 60-Asset Universe (6 Sectors)

#### 3. UMDA — Univariate Marginal Distribution Algorithm

**Type:** Univariate EDA for the discrete asset-selection outer loop

UMDA serves as the **outer loop** of a hybrid EDA: it learns which assets should be selected by maintaining a probability $\theta_i \in [0,1]$ for each asset's inclusion.

**Algorithm:**

```
Initialise: θ_i = K/N  for all i

For each generation g:
  1. Sample n_pop portfolios: draw K assets without replacement,
     weighted by θ (normalised to sum to 1)
  2. Evaluate each portfolio: solve the inner QP to get optimal weights
  3. Select top (τ · n_pop) portfolios by utility
  4. Update: θ_emp_i = frequency of asset i in top portfolios
             θ^(g+1) = (1 - smooth) · θ^(g) + smooth · θ_emp
             θ clipped to [0.01, 0.99]
```

The learned $\theta$ vector is interpretable: assets with high $\theta$ values are structurally important — the algorithm has identified that they appear consistently in the best portfolios.

**Key hyperparameters:**

| Parameter | Default | Description |
|---|---|---|
| `n_pop` | 40 | Population size |
| `n_gen` | 25–1000 | Number of outer generations |
| `tau` | 0.4 | Selection pressure (fraction kept) |
| `smooth` | 0.5 | Smoothing factor for θ update |
| `K` | 30 | Cardinality constraint |
| `LAMBDA` | 2.0 | Risk aversion |

---

#### 4. EMNA — Estimation of Multivariate Normal Algorithm

**Type:** Multivariate Gaussian EDA for the continuous weight inner loop

EMNA acts as the **inner loop** of the hybrid: given a fixed selection of K assets, it optimises the portfolio weights by fitting a multivariate Gaussian $\mathcal{N}(\boldsymbol\mu_w, \boldsymbol\Sigma_w)$ to the best weight vectors.

**Algorithm:**

```
Initialise: μ_w = (1/K, …, 1/K),  Σ_w = 0.05 · I_K

For each generation g:
  1. Sample n_pop weight vectors from N(μ_w, Σ_w)
  2. Project each sample onto the simplex {w: Σw_i=1, w_i≥0}
  3. Evaluate utility: w^T μ - λ w^T Σ w
  4. Keep the top (τ · n_pop) weight vectors
  5. Update: μ_w = mean(top),  Σ_w = cov(top) + 1e-6 I
```

**Simplex projection** uses an $O(K \log K)$ algorithm (sort + threshold) to guarantee the weight budget constraint is always satisfied.

> **Note:** In the hybrid EDA, the inner loop defaults to a QP solver (faster for convex sub-problems). EMNA remains valuable when the inner problem is itself non-convex (e.g., transaction cost penalties, non-linear constraints, regulatory lot sizes).

**Empirical result:** EMNA converges to within **< 0.01% relative gap** of the analytical QP optimum in fewer than 40 generations on the 30-asset sub-problem.

---

## Data

### Notebook 1 — Real Market Data

- **Source:** Yahoo Finance via `yfinance`
- **Universe:** 15 S&P 500 constituents across Technology, Financials, Energy, Consumer, and Healthcare sectors

| Symbol | Sector |
|---|---|
| AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA | Technology |
| JPM | Financials |
| JNJ | Healthcare |
| XOM | Energy |
| PG, KO, WMT | Consumer Staples |
| VZ | Telecom |
| GE | Industrials |

- **Period:** 3 years of daily closing prices
- **Returns:** Log-returns, annualised (×252 trading days)
- **Fat tails confirmed:** Excess kurtosis > 0 for all assets (AAPL, MSFT, GOOGL all exhibit leptokurtosis), justifying the copula approach over a pure Gaussian model.

### Notebook 2 — Simulated Universe

- **Universe:** 60 assets across 6 sectors (10 per sector): Technology, Financials, Energy, Healthcare, Consumer, Industrials
- **Data generation:** Factor model with sectoral returns

$$r_{i,t} = \beta_i \cdot f_{s(i),t} + \varepsilon_{i,t}$$

where $f_{s,t} \sim \mathcal{N}(\mu_s/252,\ \sigma_s/\sqrt{252})$ is the sector daily factor and $\varepsilon_{i,t}$ is idiosyncratic noise.

| Sector | Annual Return | Annual Vol |
|---|---|---|
| Tech | 18% | 30% |
| Financials | 10% | 22% |
| Energy | 8% | 35% |
| Healthcare | 12% | 18% |
| Consumer | 9% | 20% |
| Industrials | 10% | 24% |

- **T = 1,000** trading days (~4 years)
- Sector block structure is clearly visible in the correlation matrix.

---

## Experimental Results

### Notebook 1 — PBIL-CCPS vs Copula-EDA (K = 5, λ = 0.5)

Comparison against two benchmarks: Equal-Weight (EW) and unconstrained Markowitz minimum-variance.

| Method | Expected Return (%) | Volatility (%) | Sharpe Ratio | Active Assets |
|---|---|---|---|---|
| Equal-Weight (EW) | — | — | — | 15 |
| Min-Variance (Markowitz) | — | — | — | 15 |
| **PBIL-CCPS** | competitive | lower | higher | **5** |
| **Copula-EDA** | competitive | lowest | highest | **5** |

**Key findings:**
- Both EDAs respect the cardinality constraint exactly (K = 5 active assets).
- **Copula-EDA** consistently selects assets with lower average Kendall τ between pairs, achieving better diversification in the dependency-structure sense.
- The mean Kendall τ for Copula-EDA selected portfolios is measurably lower than for PBIL-CCPS, indicating less co-movement risk during market stress.
- **Monte Carlo simulation** (5,000 paths, 252 days): Copula-EDA achieves a better VaR(5%) and CVaR(5%) profile than PBIL-CCPS, confirming superior tail-risk management.

**Efficient frontier sweep (20 values of λ from 0.1 to 2.5):**
- Copula-EDA dominates or ties PBIL-CCPS across all 20 λ values on the Sharpe ratio axis.
- The CCMV frontier is discontinuous as expected: small changes in λ cause abrupt jumps in the optimal subset.
- Both algorithms converge stably; fitness and expected return curves are plotted across generations.

### Notebook 2 — UMDA Hybrid vs Baselines (K = 30, λ = 2.0)

| Strategy | Return | Volatility | Sharpe | Utility | Active Assets |
|---|---|---|---|---|---|
| Equal-Weight (top-K Sharpe) | baseline | higher | lower | lower | 30 |
| Markowitz long-only (no cardinality) | best | lowest | highest | highest | ~15–20 |
| **EDA Hybrid (UMDA + QP)** | near-best | near-lowest | near-highest | competitive | **30** |

**Key findings:**
- The EDA hybrid converges in **~25 outer generations** for a 60-asset universe.
- The **learned θ vector** reveals structurally important assets: those with θ → 1 appear in nearly every top portfolio regardless of λ.
- Sector allocation from EDA is diversified across all 6 sectors, with Tech and Healthcare receiving higher weights (consistent with their superior simulated Sharpe ratios).
- **Price of cardinality:** The EDA frontier lies slightly below the unconstrained Markowitz frontier — this is the cost of restricting to 30 assets instead of 60. The gap is small in practice, compensated by reduced transaction costs and simpler portfolio management.
- **EMNA vs QP:** EMNA converges to within < 0.01% of the QP solution in ≤ 40 generations, demonstrating that the continuous sub-problem is effectively solved by the EDA inner loop.

---

## Installation & Usage

### Requirements

Python 3.8+ is required. Install all dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scipy yfinance
```

### Running the notebooks

```bash
git clone https://github.com/<your-username>/EDA_Portfolio_Optimisation.git
cd EDA_Portfolio_Optimisation/notebooks
jupyter notebook
```

Open either notebook and run all cells in order. The first cell auto-installs any missing packages.

**Notebook 1** (`EDA_PBILCC_COPULA.ipynb`):
- Downloads live market data on first run (requires internet connection).
- Full run including efficient frontier sweep: approximately 2–5 minutes.

**Notebook 2** (`EDA_UMDA_EMNA.ipynb`):
- Simulates data locally; no internet required.
- Full run including frontier sweep: approximately 30–60 seconds.

### Changing key parameters

In **Notebook 1**, adjust the global variables near the top of Section 5:

```python
K      = 5      # Number of assets to select
lam    = 0.5    # Risk-aversion (higher = less risk)
```

In **Notebook 2**, adjust at the top of Section 5:

```python
K      = 30     # Cardinality
LAMBDA = 2.0    # Risk aversion
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.21 | Numerical computation, matrix operations |
| `pandas` | ≥ 1.3 | Data manipulation, descriptive statistics |
| `matplotlib` | ≥ 3.4 | Plotting, efficient frontier visualisation |
| `seaborn` | ≥ 0.11 | Correlation heatmaps |
| `scipy` | ≥ 1.7 | SLSQP optimiser, statistical tests, copula utilities |
| `yfinance` | ≥ 0.2 | Yahoo Finance data download (Notebook 1 only) |

All packages are available on PyPI and installable via `pip`.

---

## Key Visualisations

Each notebook produces a rich set of plots documenting the methodology and results:

**Notebook 1:**
- **CCMV vs Markowitz frontier** — illustrates how the cardinality constraint fragments the efficient frontier into discrete points.
- **Combinatorial explosion chart** — table + visualisation of $\binom{n}{K}$ as n grows.
- **GA infeasibility bar chart** — proportion of infeasible offspring under naive crossover.
- **Epistasis demonstration** — side-by-side bar charts showing weight redistribution after a single bit flip.
- **Real data correlation heatmap** — Pearson correlations among 15 S&P 500 assets.
- **Leptokurtosis histograms** — empirical vs Gaussian distributions for AAPL, MSFT, GOOGL.
- **PBIL convergence curves** — fitness and expected return across generations.
- **Copula Kendall τ heatmap** — rank dependency matrix used by Copula-EDA.
- **Efficient frontier comparison** — PBIL-CCPS vs Copula-EDA across 20 λ values.
- **Sharpe ratio per λ** — comparative Sharpe across the risk-return spectrum.
- **Portfolio weight allocation** — 4-panel chart showing weights for all four methods.
- **Monte Carlo return distribution** — 5,000-path bootstrap simulation with VaR markers.

**Notebook 2:**
- **Sectoral correlation heatmap** — 60×60 correlation matrix with visible block structure.
- **Mean Sharpe by sector** — bar chart of sector-level risk-adjusted returns.
- **EMNA convergence** — utility over generations vs analytical QP optimum.
- **UMDA convergence** — best utility over outer generations.
- **θ heatmap over generations** — evolution of inclusion probabilities by asset and sector.
- **Portfolio composition bar chart** — weights coloured by sector.
- **Sector allocation pie chart** — EDA portfolio diversification.
- **Risk-return scatter** — three strategies plotted against individual assets.
- **Cardinality-constrained frontier** — EDA frontier vs unconstrained Markowitz frontier.

---

## Conclusions & Future Work

### Summary

This project demonstrates that EDAs are a principled and effective approach to the CCMV problem:

- They avoid the infeasibility and epistasis problems that plague naive GAs by replacing crossover with probabilistic model learning.
- **PBIL-CCPS** (univariate) is simple, fast, and reliably converges to good cardinality-constrained portfolios.
- **Copula-EDA** (multivariate) captures non-linear tail dependencies, yielding better-diversified portfolios with lower CVaR.
- The **UMDA + QP hybrid** scales cleanly to 60 assets and learns an interpretable inclusion-probability map over the universe.
- The **price of cardinality** — the gap between the K-constrained frontier and the unconstrained frontier — is modest and practically justified by reduced transaction costs, simpler management, and improved interpretability.

### Potential Extensions

- **Replace UMDA with ECGA or BOA** — Extended Compact GA or Bayesian Optimisation Algorithm can explicitly model inter-asset dependencies (sector structure), potentially achieving faster convergence.
- **Surrogate-assisted EDA** — Use a Gaussian Process surrogate on the utility function to reduce the number of expensive QP sub-solves.
- **Multi-objective MIDEA** — Extend to a Pareto-front approach optimising Sharpe ratio and maximum drawdown simultaneously, generating a full risk profile rather than a single efficient frontier.
- **Walk-forward backtesting** — Validate out-of-sample on real historical data with rebalancing periods.
- **Transaction cost penalty** — Add a non-linear cost term to the objective, making EMNA (which handles non-convex inner problems) more valuable relative to QP.
- **Dynamic K** — Allow the cardinality to vary over time as a function of market volatility regimes.

---

## References

- Markowitz, H. (1952). *Portfolio Selection*. The Journal of Finance.
- Baluja, S. (1994). *Population-Based Incremental Learning*. CMU Technical Report.
- Larrañaga, P., & Lozano, J. A. (2002). *Estimation of Distribution Algorithms*. Kluwer Academic.
- Sklar, A. (1959). *Fonctions de répartition à n dimensions et leurs marges*.
- Chang, T. J., Meade, N., Beasley, J. E., & Sharaiha, Y. M. (2000). *Heuristics for cardinality constrained portfolio optimisation*. Computers & Operations Research.
- Mühlenbein, H., & Paass, G. (1996). *From recombination of genes to the estimation of distributions I. Binary parameters*. PPSN.

---

*EDA_Portfolio_Optimisation — Achraf Zahid & Mohamed Amine El Arbani — encadré par Pr. Abderrahim Azouani*
