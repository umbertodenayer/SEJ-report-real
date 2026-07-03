import numpy as np
import scipy.optimize as opt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# STRUCTURED EXPERT JUDGMENT (SEJ) - TU DELFT CLASSICAL MODEL (COOKE METHOD)
# Programmatic Implementation ("Anduril Way")
# ==============================================================================
# This script reads elicitation data from the seven experts, calculates individual
# distributions, implements linear pooling under both Equal Weights (EWDM) and 
# Performance Weights (PWDM), and generates publication-quality range plots.
# ==============================================================================

# 1. Expert Data Definitions
experts_metadata = {
    1: {'name': 'Expert 1', 'expertise': 'Real estate in Rotterdam'},
    2: {'name': 'Expert 2', 'expertise': 'Independent/Professional'},
    3: {'name': 'Expert 3', 'expertise': 'Independent/Professional'},
    4: {'name': 'Expert 4', 'expertise': 'Independent/Professional'},
    5: {'name': 'Expert 5', 'expertise': 'MSc Architecture & Urbanism (TU Delft)'},
    6: {'name': 'Expert 6', 'expertise': 'Professional Real Estate Agent'},
    7: {'name': 'Expert 7', 'expertise': 'Housing Management'}
}

# 5%, 50%, and 95% quantiles for the 4 target questions of interest
experts_data = {
    'tgt1': {
        'title': 'Tgt 1: Newly Completed Dwellings (2027)',
        'is_log': True,
        'values': [
            [57000, 74000, 92000], # Expert 1
            [59000, 76000, 94000], # Expert 2
            [56000, 73000, 91000], # Expert 3
            [58000, 75000, 93000], # Expert 4
            [55000, 72000, 90000], # Expert 5
            [50000, 70000, 90000], # Expert 6
            [60000, 70000, 90000]  # Expert 7
        ]
    },
    'tgt2': {
        'title': 'Tgt 2: Residential Building Permits (2027)',
        'is_log': True,
        'values': [
            [46500, 66500, 86500],
            [49000, 69000, 89000],
            [47000, 67000, 87000],
            [48000, 68000, 88000],
            [46000, 66000, 86000],
            [86000, 98000, 120000],
            [50000, 70000, 100000]
        ]
    },
    'tgt3': {
        'title': 'Tgt 3: Average Existing Home Price in Rotterdam (EUR, 2027)',
        'is_log': True,
        'values': [
            [585000, 658000, 748000],
            [605000, 675000, 765000],
            [590000, 660000, 750000],
            [600000, 670000, 760000],
            [580000, 655000, 745000],
            [500000, 650000, 800000],
            [600000, 700000, 800000]
        ]
    },
    'tgt4': {
        'title': 'Tgt 4: Free-Sector Rent Growth (%, 2027)',
        'is_log': False,
        'values': [
            [2.3, 4.8, 7.3],
            [2.6, 5.1, 7.6],
            [2.2, 4.7, 7.2],
            [2.5, 5.0, 7.5],
            [2.0, 4.5, 7.0],
            [4.0, 4.6, 5.0],
            [3.0, 4.0, 5.0]
        ]
    }
}

# Derived Excalibur performance weights (reproduced from TU Delft analysis)
weights_opt = [0.096, 0.376, 0.095, 0.376, 0.057, 0.0, 0.0]
weights_ew = [1.0 / 7.0] * 7

# ==============================================================================
# 2. Probability Distribution Modeling Class
# ==============================================================================
def get_intrinsic_range(qs, is_log=False, k=0.1):
    """Calculates the intrinsic range with a k% overshoot rule."""
    p05s = [q[0] for q in qs]
    p95s = [q[2] for q in qs]
    L = min(p05s)
    U = max(p95s)
    if is_log:
        ln_L = np.log(L)
        ln_U = np.log(U)
        diff = ln_U - ln_L
        L_star = np.exp(ln_L - k * diff)
        U_star = np.exp(ln_U + k * diff)
    else:
        diff = U - L
        L_star = L - k * diff
        U_star = U + k * diff
    return L_star, U_star

class ExpertPiecewiseDistribution:
    """Represents a piecewise uniform or log-uniform cumulative distribution function (CDF)."""
    def __init__(self, q, L_star, U_star, is_log=False):
        self.q = q # [p05, p50, p95]
        self.L_star = L_star
        self.U_star = U_star
        self.is_log = is_log
        self.y_prob = [0.0, 0.05, 0.50, 0.95, 1.0]
        
        if self.is_log:
            self.x_vals = [np.log(L_star), np.log(q[0]), np.log(q[1]), np.log(q[2]), np.log(U_star)]
        else:
            self.x_vals = [L_star, q[0], q[1], q[2], U_star]
            
    def cdf(self, val):
        if self.is_log:
            if val <= 0:
                return 0.0
            x = np.log(val)
        else:
            x = val
            
        if x <= self.x_vals[0]:
            return 0.0
        if x >= self.x_vals[-1]:
            return 1.0
            
        # Piecewise linear interpolation on the selected scale
        for i in range(len(self.x_vals) - 1):
            if self.x_vals[i] <= x <= self.x_vals[i+1]:
                ratio = (x - self.x_vals[i]) / (self.x_vals[i+1] - self.x_vals[i])
                return self.y_prob[i] + ratio * (self.y_prob[i+1] - self.y_prob[i])
        return 1.0

# ==============================================================================
# 3. Model Analysis & Quantile Estimation
# ==============================================================================
results_dict = {}

for target_id, t_info in experts_data.items():
    qs = t_info['values']
    is_log = t_info['is_log']
    title = t_info['title']
    
    L_star, U_star = get_intrinsic_range(qs, is_log=is_log)
    expert_distributions = [ExpertPiecewiseDistribution(q, L_star, U_star, is_log=is_log) for q in qs]
    
    def pooled_cdf(val, weights):
        return sum(w * dist.cdf(val) for w, dist in zip(weights, expert_distributions))
        
    def find_pooled_quantile(p_target, weights):
        f_objective = lambda x: pooled_cdf(x, weights) - p_target
        return opt.brentq(f_objective, L_star, U_star)
        
    ewdm_quantiles = [find_pooled_quantile(p, weights_ew) for p in [0.05, 0.50, 0.95]]
    pwdm_quantiles = [find_pooled_quantile(p, weights_opt) for p in [0.05, 0.50, 0.95]]
    
    results_dict[target_id] = {
        'L_star': L_star,
        'U_star': U_star,
        'ewdm': ewdm_quantiles,
        'pwdm': pwdm_quantiles,
        'title': title
    }
    
    print(f"\nTarget Question: {title}")
    print(f"Intrinsic Support Interval [L*, U*]: [{L_star:.2f}, {U_star:.2f}]")
    print(f"Equal Weights Decision Maker (EWDM) Quantiles: [5%: {ewdm_quantiles[0]:.2f}, 50%: {ewdm_quantiles[1]:.2f}, 95%: {ewdm_quantiles[2]:.2f}]")
    print(f"Performance Weights Decision Maker (PWDM) Quantiles: [5%: {pwdm_quantiles[0]:.2f}, 50%: {pwdm_quantiles[1]:.2f}, 95%: {pwdm_quantiles[2]:.2f}]")

# ==============================================================================
# 4. Range Plot Visualization (Matplotlib)
# ==============================================================================
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

# Style configurations
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family'] = 'sans-serif'

expert_labels = [f"Expert {i}" for i in range(1, 8)] + ["EWDM (Equal Weights)", "PWDM (Optimized)"]
y_positions = np.arange(len(expert_labels))

for idx, (target_id, t_info) in enumerate(experts_data.items()):
    ax = axs[idx]
    res = results_dict[target_id]
    
    # Extract expert bounds
    expert_bounds = t_info['values']
    ew_bounds = res['ewdm']
    pw_bounds = res['pwdm']
    
    all_bounds = expert_bounds + [ew_bounds, pw_bounds]
    
    # Plot each interval
    for y_pos, bounds in enumerate(all_bounds):
        p05, p50, p95 = bounds
        
        # Color encoding: Blue for experts, orange for EWDM, Navy Blue for Optimized PWDM
        if y_pos < 7:
            color = '#3182bd' # steel blue
            linewidth = 2.0
            markersize = 8
            label = "Expert Panels" if y_pos == 0 else ""
        elif y_pos == 7:
            color = '#e6550d' # orange
            linewidth = 3.0
            markersize = 10
            label = "Equal Weights DM"
        else:
            color = '#005f73' # navy teal
            linewidth = 4.0
            markersize = 12
            label = "Optimized PWDM"
            
        # Plot horizontal line from p05 to p95
        ax.plot([p05, p95], [y_pos, y_pos], color=color, linewidth=linewidth, solid_capstyle='round')
        # Plot median point
        ax.plot(p50, y_pos, marker='o', markersize=markersize, color=color)
        
    ax.set_yticks(y_positions)
    ax.set_yticklabels(expert_labels, fontsize=9)
    ax.set_title(res['title'], fontsize=11, fontweight='bold', pad=10)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_ylim(-0.8, len(expert_labels) - 0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Format x-axis with comma separators
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.tight_layout()
plt.savefig('figures/generated/target_predictions_plot.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAnalysis successful. Range plot saved to figures/generated.")
