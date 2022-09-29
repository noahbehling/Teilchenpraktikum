import matplotlib.pyplot as plt
import uproot3
import os
import numpy as np


def plot(data_zprime, data_ttbar, bins, label, feature):
    x = bins + ([*bins[1:], bins[0]] - bins)/2
    x = x[:-1]
    max_zprime = np.sum(data_zprime[1:-1])
    max_ttbar = np.sum(data_ttbar[1:-1])
    ttbar_norm = data_ttbar/max_ttbar
    zprime_norm = data_zprime/max_zprime
    plt.errorbar(x, data_zprime[1:-1]/max_zprime, yerr=np.zeros_like(data_zprime[1:-1]),
                 xerr=(bins[1]-bins[0])/2, fmt="C0_", label=r"$Z'$")
    plt.vlines(bins, ymin=zprime_norm[:-1], ymax=zprime_norm[1:], colors="C0")
    plt.errorbar(x, data_ttbar[1:-1]/max_ttbar, yerr=np.zeros_like(data_ttbar[1:-1]),
                 xerr=(bins[1]-bins[0])/2, fmt="C1_", label=r"$t\bar{t}$")
    plt.vlines(bins, ymin=ttbar_norm[:-1], ymax=ttbar_norm[1:], colors="C1")
    plt.legend()
    plt.xlim(bins[0], bins[-1])
    plt.xlabel(label)
    plt.ylabel("Density")
    plt.savefig(f"{os.getcwd()}/ATLAS/python_plots/{feature}.pdf")
    plt.clf()
    return


features = ["met_et", "del_phi", "dis3", "dis4", "dis5"]

bins = {
        "met_et": np.linspace(40000, 200000, 26),
        "del_phi": np.linspace(0, 3.12, 26),
        "dis3": np.linspace(0, 150000, 26),
        "dis4": np.linspace(150000, 3200000, 26),
        "dis5": np.linspace(-3, 3, 26)}

labels = {
         "met_et": r"$p_{T, miss.}$ [MeV]",
         "del_phi": r"$\Delta \phi$",
         "dis3": r"$m(3j)$ [MeV]",
         "dis4": r"$m(4j,\,l,\,\nu)$ [MeV]",
         "dis5": r"$\eta(4j,\,l,\,\nu)$"}

path = f"{os.getcwd()}/ATLAS/code/plots/root/"
ttbar_el = uproot3.open(path + "ttbar.el_del_phi_hist.root")["del_phi"]
zprime_el = uproot3.open(path + "zprime1000.el_del_phi_hist.root")["del_phi"]

for feat in features:
    if feat == "dis3":
        ttbar = uproot3.open(path + f"ttbar.el_{feat}_hist.root")["m_jets_pt"]
        zprime = uproot3.open(path + f"zprime1000.el_{feat}_hist.root")["m_jets_pt"]
    elif feat == "dis4":
        ttbar = uproot3.open(path + f"ttbar.el_{feat}_hist.root")["m_event"]
        zprime = uproot3.open(path + f"zprime1000.el_{feat}_hist.root")["m_event"]
    elif feat == "dis5":
        ttbar = uproot3.open(path + f"ttbar.el_{feat}_hist.root")["Eta_event"]
        zprime = uproot3.open(path + f"zprime1000.el_{feat}_hist.root")["Eta_event"]
    else:
        ttbar = uproot3.open(path + f"ttbar.el_{feat}_hist.root")[feat]
        zprime = uproot3.open(path + f"zprime1000.el_{feat}_hist.root")[feat]

    plot(zprime, ttbar, bins[feat], labels[feat], feat)
