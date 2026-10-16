import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov

# - Curve 1: mode shapes --
x = np.linspace(0, 1, 500)                  # position along the beam (x/L)

for n in [1, 2, 3, 4]:
    phi = np.sqrt(2) * np.sin(n * np.pi * x)   # Eq. 6
    plt.plot(x, phi, label=f"mode {n}")

plt.xlabel("x / L")
plt.ylabel("mode shape")
plt.legend()
plt.title("First four mode shapes")
plt.show()

# --- Curve 2: modal variance ----
modes = np.arange(1, 17)                    # modes 1 to 16

# Kelvin-Voigt: variance = 1 / (n*pi)^8      (Eq. 29)
kv = 1 / (modes * np.pi)**8
plt.semilogy(modes, kv / kv[0], "o-", label="Kelvin-Voigt")

# SLS: solve a small Lyapunov equation for each mode   (Eqs. 32-36)
r = 0.5
for Theta in [0.1, 1, 10]:
    sls = []
    for n in modes:
        K0 = (n * np.pi)**4
        A = np.array([[0, 1, 0],
                      [0, 0, -1],
                      [r * K0 / Theta, K0, -1 / Theta]])
        B = np.array([[0], [1], [0]])
        P = solve_continuous_lyapunov(A, -B @ B.T)
        sls.append(P[0, 0])                 # variance of mode n
    sls = np.array(sls)
    plt.semilogy(modes, sls / sls[0], "o-", label=f"SLS, Theta = {Theta}")

plt.xlabel("mode number n")
plt.ylabel("variance / mode-1 variance")
plt.legend()
plt.title("How much each mode moves")
plt.show()
