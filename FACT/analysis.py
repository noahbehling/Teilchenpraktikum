import numpy as np
from uncertainties import unumpy as unp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import poisson
from fact.io import read_h5py
import pandas as pd
import boost_histogram as bh

# Daten laden
sim = read_h5py("Data/gamma_test_dl3.hdf5", mode="r+", key="events")

sim_events = read_h5py(
    "Data/gamma_corsika_headers.hdf5", mode="r+", key="corsika_events",
)
sim_runs = read_h5py("Data/gamma_corsika_headers.hdf5", mode="r+", key="corsika_runs")

data_events = read_h5py("Data/open_crab_sample_dl3.hdf5", mode="r+", key="events")
data_runs = read_h5py("Data/open_crab_sample_dl3.hdf5", mode="r+", key="runs")

data_events = data_events[data_events.gamma_prediction >= 0.8]
sim_sel = sim[sim.gamma_prediction >= 0.8]

# Theta-Squared-Plot
data_events["theta_deg2"] = data_events["theta_deg"] ** 2
theta_on = data_events["theta_deg2"].values

theta_off = [0, 0, 0, 0, 0]
for i in range(1, 6):
    data_events[f"theta_deg_off_{i}2"] = data_events[f"theta_deg_off_{i}"] ** 2
    theta_off[i - 1] = data_events[f"theta_deg_off_{i}2"].values

theta_off = np.concatenate(theta_off)

H = bh.Histogram
cat = bh.axis.IntCategory([0, 1])
theta2_axis = bh.axis.Regular(40, 0, 0.3)

hist = H(theta2_axis, cat)
hist.fill(theta_on, np.ones_like(theta_on))
hist.fill(theta_off, np.zeros_like(theta_off))
cx = hist.axes[0].centers
w, xe, xi = hist.to_numpy(flow=True)
w_back, w_sig = w[1:-1, 0], w[1:-1, 1]
xe = xe[1:-1]
w = np.sum(w[1:-1, [0, 1]], axis=1)

plt.stairs(w_back / 5, xe, color="lightgray", fill=True)
# plt.errorbar(cx, w_sig, np.sqrt(w_sig), 0, label="On", fmt="+", mew=.1)
plt.plot(cx, w_sig, "+", label="On", ms=7.5)
plt.plot(cx, w_back / 5, label="Off", marker="_", ms=8, linestyle="")
plt.axvline(0.025, color="darkgray", linestyle="dashed")
plt.xlabel(r"$\theta^2$/$(°)^2$")
plt.ylabel("Counts")
plt.xlim(0, 0.3)
plt.ylim(0, 400)
plt.legend(loc=0)
plt.tight_layout()
plt.savefig("plots/theta_square.pdf")
plt.clf()

# Cut on theta^2
mask = (
    (data_events.theta_deg2 <= 0.025)
    | (data_events.theta_deg_off_12 <= 0.025)
    | (data_events.theta_deg_off_22 <= 0.025)
    | (data_events.theta_deg_off_32 <= 0.025)
    | (data_events.theta_deg_off_42 <= 0.025)
    | (data_events.theta_deg_off_52 <= 0.025)
)
data_events = data_events[mask]
sim_sel = sim_sel[
    (sim_sel["theta_deg"] <= np.sqrt(0.025))
    | (sim_sel["theta_deg_off_1"] <= np.sqrt(0.025))
    | (sim_sel["theta_deg_off_2"] <= np.sqrt(0.025))
    | (sim_sel["theta_deg_off_3"] <= np.sqrt(0.025))
    | (sim_sel["theta_deg_off_4"] <= np.sqrt(0.025))
    | (sim_sel["theta_deg_off_5"] <= np.sqrt(0.025))
]


def LiMa(N_on, N_off, alpha):
    return np.sqrt(2) * np.sqrt(
        N_on * np.log((1 + alpha) / alpha * (N_on / (N_on + N_off)))
        + N_off * np.log((1 + alpha) * N_off / (N_on + N_off))
    )


N_on = len(data_events.theta_deg2[data_events.theta_deg2 <= 0.025])
N_off = len(data_events.theta_deg2[data_events.theta_deg2 > 0.025])
print("Significance (Li & Ma):\n", LiMa(N_on, N_off, 0.2))

# Energy Migration

pred_axis = bh.axis.Regular(10, 500, 15e3, transform=bh.axis.transform.log)
true_axis = bh.axis.Regular(5, 500, 15e3, transform=bh.axis.transform.log)

E_true = sim_sel.corsika_event_header_total_energy.values
E_pred = sim_sel.gamma_energy_prediction.values

E_hist = H(pred_axis, true_axis)
E_hist.fill(E_pred, E_true)


def plothist2d(h):
    return plt.pcolormesh(*h.axes.edges.T, h.view().T)


w, xe, xi = E_hist.to_numpy(flow=True)
print("Matrix: ", w)
w = w / np.sum(w, axis=0)

plt.matshow(w)
plt.xlabel("True energy / GeV")
plt.ylabel("Predicted energy / GeV")
plt.colorbar()
plt.savefig("plots/Matrix.pdf")
plt.close()

# Unfolding with naive SVD
print("Matrix after normalising: ", w)
w_inv = np.linalg.pinv(w)

g_hist = H(pred_axis)
g_hist.fill(data_events.gamma_energy_prediction[data_events["theta_deg2"] <= 0.025])
g, g_bins = g_hist.to_numpy(flow=True)
g_unc = unp.uarray(g, np.sqrt(g))

b_hist = H(pred_axis)
b_hist.fill(data_events.gamma_energy_prediction[data_events["theta_deg2"] > 0.025])
b, b_bins = b_hist.to_numpy(flow=True)
b = b / 5
b_unc = unp.uarray(b, np.sqrt(b))

diff = g - b
diff_unc = g_unc - b_unc
fNSVD = w_inv @ diff
fNSVD = fNSVD[1:-1]
fNSVD_unc = w_inv.dot(diff_unc)[1:-1]
print("fNSVD: ", fNSVD_unc)
xi = xi[1:-1]
xerr = np.array([xi[i] - xi[i - 1] for i in range(1, len(xi))])
xerr = xerr / 2
plt.errorbar(
    E_hist.axes[1].centers, fNSVD, xerr=xerr, yerr=unp.std_devs(fNSVD_unc), fmt="C0+"
)
plt.semilogx()
plt.semilogy()
plt.xlabel("E$_\mathrm{est}$ / GeV")
plt.ylabel("Number of Events")
plt.title("SVD Unfolding")
plt.tight_layout()
plt.savefig("plots/NSVD.pdf")
plt.clf()

# Poisson-Likelihood-Entfaltung


def NLL(f, A, b, g):
    lam = A @ f + b
    return -poisson.logpmf(g, lam).sum()


full_hist = H(true_axis)
full_hist.fill(E_pred)
f, f_edges = full_hist.to_numpy(flow=True)

Bounds = [(1, 100000)] * len(f)
est = minimize(NLL, f, args=(w, b, g), method="L-BFGS-B", bounds=Bounds)

fPoisson = est["x"][1:-1]

hess = est["hess_inv"].todense()
diag_cov = np.diag(hess)
std_devs = np.sqrt(diag_cov)[1:-1]
fPoisson_unc = unp.uarray(fPoisson, std_devs)
print("fPoisson: ", fPoisson_unc)

plt.errorbar(E_hist.axes[1].centers, fPoisson, xerr=xerr, yerr=std_devs, fmt="C0+")
plt.semilogx()
plt.semilogy()
plt.xlabel("E$_\mathrm{est}$ / GeV")
plt.ylabel("Number of Events")
plt.title("Poisson-Likelihood Unfolding")
plt.tight_layout()
plt.savefig("plots/PL.pdf")
plt.clf()


# Comparison plot
plt.errorbar(
    E_hist.axes[1].centers,
    fNSVD,
    xerr=xerr,
    yerr=unp.std_devs(fNSVD_unc),
    fmt="x",
    label="Naive SVD Unfolding",
    alpha=0.8,
)
plt.errorbar(
    E_hist.axes[1].centers,
    fPoisson,
    xerr=xerr,
    yerr=std_devs,
    fmt="x",
    label="Poisson-Likelihood Unfolding",
    alpha=0.8,
)
plt.semilogx()
plt.semilogy()
plt.xlabel("E$_\mathrm{est}$ / GeV")
plt.ylabel("Number of Events")
plt.title("Comparison of the Unfolding methods")
plt.legend(loc=0)
plt.tight_layout()
plt.savefig("plots/Unfolding_compare.pdf")
plt.clf()

# Acceptance Correction
A = np.pi * 27000 ** 2
t_obs = data_runs.ontime.sum()
dE = xerr * 2 * 10 ** -3

sim_hist = H(true_axis)
sim_hist.fill(sim_events.total_energy)
N_sim, Ne = sim_hist.to_numpy()
N_sel = f[1:-1]

A_eff = N_sel / N_sim * A / 0.7
print("Effective Detection Area: ", A_eff)


def flux(f, A, dE, t):
    return f / (A * dE * t)


flux_NSVD = flux(fNSVD_unc, A_eff, dE, t_obs)
flux_Poisson = flux(fPoisson_unc, A_eff, dE, t_obs)

print("NSVD flux: ", flux_NSVD)
print("Poisson flux: ", flux_Poisson)

plt.errorbar(
    E_hist.axes[1].centers * 0.99,
    unp.nominal_values(flux_NSVD),
    xerr=xerr,
    yerr=unp.std_devs(flux_NSVD),
    fmt="x",
    label="Naive SVD",
    alpha=0.8,
)
plt.errorbar(
    E_hist.axes[1].centers * 1.01,
    unp.nominal_values(flux_Poisson),
    xerr=xerr,
    yerr=unp.std_devs(flux_Poisson),
    fmt="x",
    label="Poisson-Likelihood",
    alpha=0.8,
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Energy$_\mathrm{true}$ /GeV")
plt.ylabel(
    r"$\frac{\mathrm{d}N}{\mathrm{d}E \, \mathrm{d}A \, t_{obs}}$ [TeV$^{-1}$ cm$^{-2}$ s$^{-1}$]"
)
plt.legend()
plt.tight_layout()
plt.savefig("plots/flux.pdf")
plt.clf()


def func(x, a, b, c, d):
    return a * (x / b) ** (-c + d * np.log(x / b))


x = np.linspace(450, 17000, 10000)
phi_MAGIC = func(x, 3.23 * 10 ** (-11), 1000, 2.47, -0.24)
phi_HEGRA = func(x, 2.83 * 10 ** (-11), 1000, 2.62, 0)
plt.plot(x, phi_MAGIC, "k-", label="MAGIC")
plt.plot(x, phi_HEGRA, "g-", label="HEGRA")
plt.errorbar(
    E_hist.axes[1].centers,
    unp.nominal_values(flux_NSVD),
    xerr=xerr,
    yerr=unp.std_devs(flux_NSVD),
    fmt="x",
    label="Naive SVD",
    alpha=0.8,
)
plt.errorbar(
    E_hist.axes[1].centers,
    unp.nominal_values(flux_Poisson),
    xerr=xerr,
    yerr=unp.std_devs(flux_Poisson),
    fmt="x",
    label="Poisson-Likelihood",
    alpha=0.8,
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Energy$_\mathrm{true}$ /GeV")
plt.ylabel(
    r"$\frac{\mathrm{d}N}{\mathrm{d}E \, \mathrm{d}A \, t_{obs}}$ [TeV$^{-1}$ cm$^{-2}$ s$^{-1}$]"
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Energy$_\mathrm{true}$ /GeV")
plt.ylabel(
    r"$\frac{\mathrm{d}N}{\mathrm{d}E \, \mathrm{d}A \, t_{obs}}$ [TeV$^{-1}$ cm$^{-2}$ s$^{-1}$]"
)
plt.legend()
plt.tight_layout()
plt.savefig("plots/flux_compare.pdf")
plt.close()
