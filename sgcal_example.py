# -*- coding: utf-8 -*-
"""
Simple example with the solvers pgd-l0, pgd-l2 and vanilla.

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from sgcal_ambient import CalibrateDiffuse

M = 50          # number of sensors
wavelength = 2  # wavelength [m]
var_n = 1e-3    # noise variance

# %% Generate data
np.random.seed(212)

pos = 1*np.random.rand(M, 2)  # sensor positions in the plane
a_gt = np.random.rand(M) + 0.5 + 1j*np.random.rand(M) - 0.5j  # sensor gains

# model of ambient noise covariance
dist = squareform(pdist(pos))
S = np.sinc(2*np.pi/wavelength*dist)

# create measured covariance for infinite number of snapshots
N_ambient = np.diag(a_gt) @ S @ np.diag(a_gt.conj())
N_sensor = var_n * np.eye(M)
R = N_ambient + N_sensor

# %% calibration solvers
cd_solver0 = CalibrateDiffuse(R, S)  # pgd-l0
cd_solver0.pgd_l0()

cd_solver1 = CalibrateDiffuse(R, S)  # pgd-l1
cd_solver1.pgd_l1(reg_l1=0.01)

cd_solver2 = CalibrateDiffuse(R, S)  # vanilla
cd_solver2.vanilla()

# scale the solutions
cd_solver0.scale_to(a_gt[:, None])
cd_solver1.scale_to(a_gt[:, None])
cd_solver2.scale_to(a_gt[:, None]) 

print("SNR:", 10*np.log10(np.linalg.norm(N_ambient)/np.linalg.norm(N_sensor)))
print("rmse vanilla:", np.linalg.norm(a_gt - cd_solver2.a_est.T))
print("rmse pgd-l0:", np.linalg.norm(a_gt - cd_solver0.a_est.T))
print("rmse pgd-l1:", np.linalg.norm(a_gt - cd_solver1.a_est.T))

# %% Plot results
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7, 4))

for ax_, title_, solver_ in zip([ax0, ax1], ["pgd-l0", "pgd-l1"],
                                [cd_solver0, cd_solver1]):

    ax_.scatter(a_gt.real, a_gt.imag, label="ground truth", s=5, c="black")
    ax_.scatter(cd_solver2.a_est.real, cd_solver2.a_est.imag, marker="s",
                facecolors="none", edgecolors="orange", label="svd")
    ax_.scatter(solver_.a_est.real, solver_.a_est.imag,
                facecolors="none", edgecolors="green",
                label=title_)

    ax_.grid()
    ax_.axis("scaled")
    ax_.set_xlim(-1.8, 1.8)
    ax_.set_ylim(-1.8, 1.8)
    ax_.set_title(title_)
    ax_.legend()

plt.suptitle("Gain/phase calibration of sensor array from ambient noise by\n"
             "covariance measurement fitting - simple example.")
plt.tight_layout()
