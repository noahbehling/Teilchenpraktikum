from fact.io import read_h5py, to_h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from uncertainties import unumpy as unp
from uncertainties import ufloat

#READ DATA
crab_events = read_h5py('data/open_crab_sample_dl3.hdf5', key='events')
crab_runs = read_h5py('data/open_crab_sample_dl3.hdf5', key='runs')
gamma_events = read_h5py('data/gamma_test_dl3.hdf5', key='events')
corsika_events = read_h5py('data/gamma_corsika_headers.hdf5', key='corsika_events')
corsika_runs = read_h5py('data/gamma_corsika_headers.hdf5', key='corsika_runs')

#SELECTION
crab_events_pred = crab_events[crab_events['gamma_prediction'] > 0.8]
gammas_pred = gamma_events[gamma_events['gamma_prediction'] > 0.8]

lim_theta = np.sqrt(0.025)

crab_events_sel = crab_events_pred[crab_events_pred['theta_deg'] < lim_theta]
crab_events_sel_1 = crab_events_pred[crab_events_pred['theta_deg_off_1'] < lim_theta]
crab_events_sel_2 = crab_events_pred[crab_events_pred['theta_deg_off_2'] < lim_theta]
crab_events_sel_3 = crab_events_pred[crab_events_pred['theta_deg_off_3'] < lim_theta]
crab_events_sel_4 = crab_events_pred[crab_events_pred['theta_deg_off_4'] < lim_theta]
crab_events_sel_5 = crab_events_pred[crab_events_pred['theta_deg_off_5'] < lim_theta]
gammas_sel = gammas_pred[gammas_pred['theta_deg'] < lim_theta]

bkg = pd.concat([crab_events_sel_1, crab_events_sel_2, crab_events_sel_3, crab_events_sel_4, crab_events_sel_5])


lim_theta_plots = np.sqrt(0.3)

crab_events_pred_1 = crab_events_pred[crab_events_pred['theta_deg_off_1'] < lim_theta_plots]
crab_events_pred_2 = crab_events_pred[crab_events_pred['theta_deg_off_2'] < lim_theta_plots]
crab_events_pred_3 = crab_events_pred[crab_events_pred['theta_deg_off_3'] < lim_theta_plots]
crab_events_pred_4 = crab_events_pred[crab_events_pred['theta_deg_off_4'] < lim_theta_plots]
crab_events_pred_5 = crab_events_pred[crab_events_pred['theta_deg_off_5'] < lim_theta_plots]
crab_events_sel_plot = crab_events_pred[crab_events_pred['theta_deg'] < lim_theta_plots]

theta_deg_off = []
for i in [1, 2, 3, 4, 5]:
    exec('x = crab_events_pred_{}.theta_deg_off_{}.values'.format(i, i))
    for k in x:
        theta_deg_off.append(k)

crab_events_sel_on = np.array(crab_events_sel_plot['theta_deg'].values)

#
plt.hist((crab_events_sel_on)**2, bins =40, histtype='step', color='deeppink', label='On-events')
plt.hist(np.array(theta_deg_off)**2, bins=40, histtype='step', color='dodgerblue', label='Off-events', weights=np.array([0.2 for el in theta_deg_off]))
plt.vlines(0.025, color='black', linestyle='-.', ymin=0, ymax=400, label=r'$\theta^{2}-cut$')
plt.xlabel(r'$\theta^2$ / $\deg^2$')
plt.legend()
plt.tight_layout()
plt.text(0.05, 350,
         r'''Source: Crab, $t_\mathrm{{obs}}$ = 17.7h
$N_\mathrm{{on}}$ = {non}, $N_\mathrm{{off}}$ = {noff}, $\alpha$ = 0.2'''.format(non=len(crab_events_sel), noff=len(bkg)))
plt.savefig('plots/On_Off.pdf')
plt.close()
#print(len(crab_events_sel_1.theta_deg_off_1.values))
#DETECTION SIGNIFICANCE
alpha = 0.2
n_on = len(crab_events_sel)
n_off = len(bkg)
sum1 = n_on * np.log((1 + alpha) / alpha * (n_on / (n_on + n_off)))
sum2 = n_off * np.log((1 + alpha) * (n_off / (n_on + n_off)))
S = np.sqrt(2) * np.sqrt(sum1 + sum2)
print("Detektions-Signifikanz: ", S)

#ENERGY MIGRATION
gamma_E_pred = gammas_sel['gamma_energy_prediction']
corsika_total_E = gammas_sel['corsika_event_header_total_energy']
    #BINNING
max_bin = max(max(gamma_E_pred), max(corsika_total_E))
min_bin = min(min(gamma_E_pred), min(corsika_total_E))

if max_bin<= max(np.logspace(np.log10(500), np.log10(20e3), 11)):
    next_bin = 12
    bins_1 = np.ones(next_bin)
else:
    next_bin = 13
    bins_1 = np.ones(next_bin)
    bins_1[-1] = 50e3
bins_1[0] = 0
for i in range(1, next_bin-1, 1):
    bins_1[i] = np.logspace(np.log10(500), np.log10(15e3), 11)[i-1]

bins1 = np.ones(8)
bins1[0] = 0
for i in range(1, 7, 1):
    bins1[i] = np.logspace(np.log10(500), np.log10(15e3), 6)[i-1]
bins1[-1] = 50e3

#PLOTTING MATRIX
plt.figure(constrained_layout=True)
matrix, xedge, yedge = np.histogram2d(gamma_E_pred, corsika_total_E, bins=[bins_1, bins1])

matrix = matrix / np.sum(matrix, axis=0)
# print(matrix)
plt.matshow(matrix)
plt.xlabel('corsika')
plt.ylabel('gamma prediction')
plt.colorbar()
plt.savefig('plots/Matrix.pdf')
plt.close()
#PLOT ENERGY DISTRIBUTION: DOESNT WORK CURRENTLY
# plt.subplot(1, 2, 1)
a = np.full_like(bkg['gamma_energy_prediction'], 0.2)
b, bins_b, p = plt.hist(bkg['gamma_energy_prediction'], bins=xedge, weights=a, color='b', label='Off-Events')

# # plt.xscale('log')
# # plt.yscale('log')
# # plt.xlim(1e2, 1e5)
# # plt.xlabel('E_Est / GeV')
# # plt.ylabel('Number of Events')
# # plt.legend()
# # plt.title('Energy Distribution')
# # plt.subplot(1, 2, 2)
g, bins_g, p = plt.hist(crab_events_sel['gamma_energy_prediction'], bins=xedge, color='c', label='On-Events')

# # plt.xscale('log')
# # plt.yscale('log')
# # plt.xlim(1e2, 1e5)
# # plt.xlabel('E_Est / GeV')
# # plt.legend()
# # plt.title('Energy Distribution')
# # plt.savefig('plots/E_verteilung.pdf')
plt.close()
#UNFOLDING
#SVD UNFOLDING
pseudo_inv = unp.ulinalg.pinv(matrix)
print("Pseudo-Inverse: ",pseudo_inv)
g_unc = np.array([ufloat(x, np.sqrt(x)) for x in g])
b_unc = np.array([ufloat(x, np.sqrt(x)) for x in b])
#START UNFOLDING HERE
ev = g-b
ev_unc = g_unc - b_unc
fNSVD = pseudo_inv@ev
fNSVD_unc = pseudo_inv.dot(ev_unc)[1:-1]
print(fNSVD)
print(fNSVD_unc)
fNSVD = fNSVD[1:-1]
print('fNSVD: ', fNSVD)
#
xpos = [yedge[i] - (yedge[i] - yedge[i-1])/2 for i in range(1, len(yedge))]
xpos = xpos[1:-1]
xerr = [(yedge[i] - yedge[i-1])/2 for i in range(2, len(yedge)-1)]


plt.errorbar(xpos, [x.nominal_value for x in fNSVD_unc], xerr=xerr, yerr=[x.std_dev for x in fNSVD_unc], fmt='bx')
plt.xlim(400, 1.6e4)
plt.ylim(4, 300)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('E$_\mathrm{est}$ / GeV')
plt.ylabel('Number of Events')
plt.title('SVD Unfolding')
plt.tight_layout()
plt.savefig('plots/NSVD.pdf')
plt.close()

#POISSON LIKELIHOOD UNFOLDING
def PLU(f, A, b, g):
    lam = A@f + b
    return -poisson.logpmf(g, lam).sum()

f, yedges = np.histogram(gamma_E_pred, bins=yedge)
Bounds = [(1, 100000)]*len(f)

estimator = minimize(PLU, f+100, args=(matrix, b, g), method='L-BFGS-B', bounds=Bounds)
fLike = estimator['x']
fLike_plot = estimator['x'][1:-1]

Hesse_inv = estimator['hess_inv'].todense()
print(fLike_plot)
print('cov matrix: ', Hesse_inv)

diag_cov = np.diag(Hesse_inv)
diag_cov_sqrt = np.sqrt(diag_cov)
print(diag_cov_sqrt[1:-1])
std_devs = diag_cov_sqrt[1:-1]

plt.errorbar(xpos, fLike_plot, xerr=xerr, yerr=std_devs, fmt='rx')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('E$_\mathrm{est}$ / GeV')
plt.ylabel('Number of Events')
plt.title('Poisson Likelihood Unfolding')
plt.tight_layout()
plt.savefig('plots/Unfolding_2.pdf')
plt.close()

plt.errorbar(xpos, fNSVD, xerr=xerr, yerr=[x.std_dev for x in fNSVD_unc], fmt='bx', label="Naive SVD Unfolding")
plt.errorbar(xpos, fLike_plot, xerr=xerr, yerr=std_devs, fmt='rx', label="Poisson Likelihood Unfolding")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('E$_\mathrm{est}$ / GeV')
plt.ylabel('Number of Events')
plt.title('Unfolding')
plt.legend()
plt.tight_layout()
plt.savefig('plots/Unfolding_compare.pdf')
plt.close()

#RANDOM NUMBERS WITH MULTIVARIATE NORMAL
fLike_Var = np.random.multivariate_normal(fLike, Hesse_inv, size=10000)
fLike_Var = fLike_Var[:,1:-1]
fLike_Var[:, 0]
std_devs = np.ones(5)
for i in range(5):
    std_devs[i] = np.std(fLike_Var[:, i])
print("Standard Deviations: ", std_devs)

def flux(f, A, dE, t):
    return f / (A * dE * t)

t_obs = crab_runs['ontime'].sum()

Delta_E = np.diff(yedge[1:-1]*10**(-3))

#DETECTOR AREA
A = np.pi * 27000**2
# N_sel / N_dim
hist_sel, yedges = np.histogram(gamma_E_pred, bins=yedge)
hist_sim, yedges = np.histogram(corsika_events['total_energy'], bins=yedge)

A_eff = hist_sel[1:-1] / hist_sim[1:-1] * A / 0.7
print('A_eff: ', A_eff)

#CALC OF FLUX
phi_NSVD = fNSVD_unc/(A_eff * Delta_E * t_obs)
phi_Like = fLike_plot/(A_eff * Delta_E * t_obs)

#MEAN AND DEVIATION
phi_Like_Var = fLike_Var/(A_eff * Delta_E * t_obs)
mean = phi_Like_Var.mean(axis=0)
std = phi_Like_Var.std(axis=0)
print('Phi_NSVD:', phi_NSVD)
print('Phi_Like:', phi_Like)
print('Mean:',mean)
print('Std:',std)

plt.errorbar(xpos, [x.nominal_value for x in phi_NSVD], xerr=xerr, yerr=[x.std_dev for x in phi_NSVD], fmt ='rx', label='Naive SVD')
plt.errorbar(xpos, phi_Like, xerr=xerr, yerr=std, fmt ='bx', label='Likelihood')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energy$_\mathrm{true}$ /GeV')
plt.ylabel(r'$\frac{\mathrm{d}N}{\mathrm{d}E \, \mathrm{d}A \, t_{obs}}$ [TeV$^{-1}$ cm$^{-2}$ s$^{-1}$]')
plt.legend()
plt.tight_layout()
plt.savefig('plots/flux.pdf')
plt.close()


def func(x, a, b, c, d):
     return a * (x/b) ** (-c + d * np.log(x/b))

x = np.linspace(450, 17000, 10000)
phi_MAGIC = func(x, 3.23*10**(-11), 1000, 2.47,-0.24)
phi_HEGRA = func(x, 2.83*10**(-11), 1000, 2.62, 0)
plt.plot(x, phi_MAGIC, 'k-', label='MAGIC')
plt.plot(x, phi_HEGRA, 'g-', label='HEGRA')
plt.errorbar(xpos, mean, yerr = std, xerr=xerr, fmt='bx', label='Likelihood')
#plt.fill_between(yedge[1:-1],mean-std,mean+std,facecolor='b',alpha=0.2, label='$1 \sigma$-Umgebung')
#plt.fill_betweenx(mean, yedge[1:-1]-[(yedge[i] - yedge[i-1])/2 for i in range(1,len(yedge)-1)], yedge[1:-1]+[(yedge[i] - yedge[i-1])/2 for i in range(1,len(yedge)-1)],facecolor='b',alpha=0.2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energy$_\mathrm{true}$ /GeV')
plt.ylabel(r'$\frac{\mathrm{d}N}{\mathrm{d}E \, \mathrm{d}A \, t_{obs}}$ [TeV$^{-1}$ cm$^{-2}$ s$^{-1}$]')
plt.legend()
plt.tight_layout()
plt.savefig('plots/flux_compare.pdf')
plt.close()

print("PROGRAM SUCCEEDED")
