import numpy as np
import matplotlib.pyplot as plt

# ================== YOUR LAW'S DEFINED CONSTANTS ==================
# From your First-Principles Derivation
THEORETICAL_GAMMA = 6.87
THEORETICAL_N = 2.65e-34

# ================== EXPERIMENTAL DATA ==================
m_u = np.array([720, 840, 3600, 615, 1030, 25000])
r_nm = np.array([0.50, 0.55, 0.80, 1.00, 1.10, 2.00])
p_exp = np.array([0.106, 0.104, 0.068, 0.040, 0.032, 0.015])

M_kg = m_u * 1.660539e-27 # Convert unified atomic mass units to kg
R_m = r_nm * 1e-9         # Convert nm to meters

# ================== YOUR LAW'S FORMULA ==================
def ukachi_law_theoretical(M, R, N_const, gamma_const):
    # The formula as described: P(M,R) = N / (pi * e * R^4 * log10(M/1kg)^gamma)
    # Assuming M is already in kg, M/1kg becomes just M.
    # Using abs(np.log10(M)) to handle potential negative log values in the power.
    return N_const / (np.pi * np.e * R**4 * (np.abs(np.log10(M)) ** gamma_const))

# ================== CALCULATE PREDICTED P VALUES (for experimental points) ==================
P_predicted = ukachi_law_theoretical(M_kg, R_m, THEORETICAL_N, THEORETICAL_GAMMA)

# ================== DISPLAY RESULTS AND COMPARISON ==================
print("="*80)
print("VALIDATION OF UKACHI LAW WITH THEORETICAL CONSTANTS")
print("="*80)
print(f"Theoretical γ = {THEORETICAL_GAMMA}")
print(f"Theoretical N = {THEORETICAL_N:.2e} m⁴\n")

print(f"{'Molecule':<10} {'P_exp':<10} {'P_predicted':<15} {'Absolute Error':<16} {'Percentage Error':<18}")
print("-"*80)
for i in range(len(m_u)):
    abs_error = np.abs(p_exp[i] - P_predicted[i])
    if p_exp[i] != 0:
        percent_error = (abs_error / p_exp[i]) * 100
    else:
        percent_error = np.nan

    print(f"{['C60', 'C70', 'C3600', 'C615', 'C1030', 'C25000'][i]:<10} {p_exp[i]:<10.3f} {P_predicted[i]:<15.3e} {abs_error:<16.3e} {percent_error:<18.1f}%")

print("\n" + "="*80)

# ================== PLOT COMPARISON (Extended Range) ==================
plt.figure(figsize=(14,9))

# Define a very wide mass range to show the universal scaling
# From very quantum (smaller than experimental) to macroscopic
# Adjusted M_plot_universal range to avoid M=1kg, which causes log10(M)=0 and division by zero
M_plot_universal = np.logspace(-30, np.log10(0.9999999), 1000) # Example: from 10^-30 kg to slightly less than 1 kg
R_avg_for_plot = np.mean(R_m) # Use average R for the universal curve
P_plot_universal_smooth = ukachi_law_theoretical(M_plot_universal, R_avg_for_plot, THEORETICAL_N, THEORETICAL_GAMMA)

# Plot the smooth theoretical curve across the universal range
plt.loglog(M_plot_universal, P_plot_universal_smooth, '-', color='blue', linewidth=4, alpha=0.7,
           label=f'Ukachi Law (Universal Scaling, γ={THEORETICAL_GAMMA}, N={THEORETICAL_N:.2e})', zorder=5)

# Plot the experimental data points
plt.loglog(M_kg, p_exp, 'o', color='red', markersize=14, markeredgecolor='black',
           label='Experimental Data (Purely Quantum Regime)', zorder=10)

# Highlight the theoretical prediction for the experimental points specifically
plt.loglog(M_kg, P_predicted, 's', color='lime', markersize=12, markeredgecolor='green',
           label='Theoretical Prediction (at Experimental Points)', zorder=11)

# Add annotations to explain the plot
plt.text(M_kg.max() * 5, p_exp.min() * 0.0001, 'Region of "Quantum State Fell" / Classical Regime',
         fontsize=12, color='darkred', ha='left', va='center', rotation=0)

plt.annotate('Experimental Window (High Visibility)',
             xy=(M_kg.min(), p_exp.max()), xytext=(M_kg.min() * 1e-2, p_exp.max() * 0.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             fontsize=12, color='darkgreen', ha='left', va='bottom')

plt.xlabel("Mass M (kg)", fontsize=14)
plt.ylabel("Interference Visibility P", fontsize=14)
plt.title(f"Ukachi Law: Universal Scaling (Theoretical γ={THEORETICAL_GAMMA}, N={THEORETICAL_N:.2e})", fontsize=18)
plt.legend(fontsize=12, loc='lower left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("Ukachi_Law_Universal_Validation.png", dpi=300, bbox_inches='tight')
print("\nPlot saved as Ukachi_Law_Universal_Validation.png — OPEN IT NOW!")
plt.show()
