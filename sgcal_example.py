# -*- coding: utf-8 -*-
"""
Simple example with the solvers pgd-l0, pgd-l2 and vanilla from the paper
"Gain and phase calibration of sensor arrays from ambient noise by
cross-spectral measurements fitting", Journal of the Acoustical Society of
America.

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
from scipy.spatial.distance import pdist, squareform

from sgcal_ambient import CalibrateDiffuse, randnc, corr

# %% Generate data
np.random.seed(212)

M = 50          # number of sensors
f0 = 300        # frequency [Hz]
c0 = 500        # velocity [m/s]
var_n = 1e-3    # noise variance

# define some sensor positions & distances in the plane
pos = 1*np.random.rand(M, 2)

# define some sensor sensitivities
a_gt = 1 + 1*(0.2*(np.random.randn(M) + 2*1j*np.random.randn(M)))
a_gt /= a_gt.mean()  # hyp : E{a} = 1

# model of ambient noise covariance
dist = squareform(pdist(pos))
k = 2*np.pi*f0/c0
S = np.sinc(k*dist)

# create measured covariance for infinite number of snapshots
N_ambient = np.diag(a_gt) @ S @ np.diag(a_gt.conj())
N_sensor = var_n * np.eye(M)
R = N_ambient + N_sensor

# %% calibration solvers

cd_solver0 = CalibrateDiffuse(R, S)  # pgd-l0
cd_solver0.solve_l0(step="opt")

cd_solver1 = CalibrateDiffuse(R, S)  # pgd-l1
cd_solver1.solve_relaxed(reg_l1=0.01, step="opt")

cd_solver2 = CalibrateDiffuse(R, S)  # vanilla
cd_solver2.solve_svd()

# scale the solutions
cd_solver0.scale_to(a_gt[:, None])
cd_solver1.scale_to(a_gt[:, None])
cd_solver2.scale_to(a_gt[:, None]) 

snr = 10*np.log10(np.linalg.norm(N_ambient)/np.linalg.norm(N_sensor))
print("SNR = ", snr)
print("correlation svd :", np.abs(corr(a_gt, cd_solver2.a_est)))
print("correlation l0  :", np.abs(corr(a_gt, cd_solver0.a_est)))
print("correlation l1  :", np.abs(corr(a_gt, cd_solver1.a_est)))
print("rmse svd :", np.linalg.norm(a_gt - cd_solver2.a_est.T))
print("rmse l0  :", np.linalg.norm(a_gt - cd_solver0.a_est.T))
print("rmse l1  :", np.linalg.norm(a_gt - cd_solver1.a_est.T))

# %% Plot results

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7, 4))

ax0.scatter(a_gt.real, a_gt.imag, label="ground truth")
ax0.scatter(cd_solver0.a_est.real, cd_solver0.a_est.imag,
            facecolors="none", edgecolors="blue",
            label="proximal")

segs = [[[ae.real, ae.imag], [ag.real, ag.imag]]
        for ae, ag in zip(cd_solver0.a_est, a_gt)]

line_segments = LineCollection(segs, linestyle='solid',
                               color="blue", linewidth=0.5)
ax0.add_collection(line_segments)

ax0.scatter(cd_solver2.a_est.real, cd_solver2.a_est.imag, marker="s",
            facecolors="none", edgecolors="orange", label="svd")

ax0.axis("scaled")
ax0.set_xlim(-1.8, 1.8)
ax0.set_ylim(-1.8, 1.8)
ax0.set_title("pgd-l0")

ax0.legend()

ax1.scatter(a_gt.real, a_gt.imag, label="ground truth")
ax1.scatter(cd_solver1.a_est.real, cd_solver1.a_est.imag,
            facecolors="none", edgecolors="blue",
            label="proximal")

segs = [[[ae.real, ae.imag], [ag.real, ag.imag]]
        for ae, ag in zip(cd_solver1.a_est, a_gt)]

line_segments = LineCollection(segs, linestyle='solid',
                               color="blue", linewidth=0.5)
ax1.add_collection(line_segments)

ax1.scatter(cd_solver2.a_est.real, cd_solver2.a_est.imag, marker="s",
            facecolors="none", edgecolors="orange", label="svd")

ell = Ellipse([1, 0], width=1*0.2, height=1*2)
ax0.add_artist(ell)
ell.set_alpha(0.3)
ell.set_facecolor([0.5, 0.5, 0.5])
ell = Ellipse([1, 0], width=1*0.2, height=1*2)
ax1.add_artist(ell)
ell.set_alpha(0.3)
ell.set_facecolor([0.5, 0.5, 0.5])

ax1.axis("scaled")
ax1.set_xlim(-1.8, 1.8)
ax1.set_ylim(-1.8, 1.8)
ax1.set_title("pgd-l1")

plt.tight_layout()
