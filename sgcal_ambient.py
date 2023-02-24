# -*- coding: utf-8 -*-

# Copyright (c) 2023 Charles Vanwynsberghe
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

class CalibrateDiffuse:
    def __init__(self, Y, S):
        """
        Sensor gain and phase calibration in ambient noise with covariance
        readings, with 3 implemented solvers: vanilla, pgd-l1 and pgd-l0.

        Parameters
        ----------
        Y : measured covariance matrix
        S : modeled covariance matrix in the diffuse field

        """
        self.Y = Y
        self.S = S
        self.M = self.Y.shape[0]
        self.step_opt = 2 / np.max(np.abs(self.S)**2)  # max stable step

    def _get_a_est(self, normed=False):
        """
        Get final gain vector (principal eigenvector of C).

        """
        self.a_est, s, _ = np.linalg.svd((self.C_ + self.C_.T.conj())/2)
        self.a_est = self.a_est[:, 0]
        self.a_est *= s[0]**0.5
        self.a_est = self.a_est[:, None]
        if normed:
            self.a_est /= self.a_est.mean()

    def pgd_l1(self, reg_l1=0.5, step="opt", n_it=10000, a_init=None,
               normed=False, save_iters=False, verbose=True):
        """
        Proximal gradient descent solver with trace penalty on C.

        Parameters
        ----------
        reg_l1 : regularization parameter
        step : gradient step
        n_it : max number of iterations
        a_init : gains at initialization
        normed : indicate if estimated gain vector should be normalized
        save_iters : keep all iterations
        verbose : helper message

        """
        if step == "opt":
            step = self.step_opt

        if a_init is None:
            self.a_init = np.ones((self.M, 1), np.complex)
        else:
            self.a_init = a_init

        if save_iters is True:
            self.a_list = []

        self.C_ = self.a_init @ self.a_init.T.conj()
        self.rk_C_ = np.zeros((n_it))
        for n_ in range(n_it):
            C_old = self.C_
            self.Z_ = self.C_ + step/2*self.S.conj()*(self.Y - self.S*self.C_)
            self.C_, self.rk_C_[n_] = prox_st((self.Z_ + self.Z_.T.conj())/2,
                                                 step*reg_l1)

            self._get_a_est(normed=normed)
            if save_iters is True:
                self.a_list.append(self.a_est)

            if np.linalg.norm(C_old - self.C_) < 1e-6:
                self.rk_C_ = self.rk_C_[0:n_+1]
                if save_iters is True:
                    self.a_list = np.array(self.a_list).squeeze()
                if verbose:
                    print(f"pgd-l1 done, {n_} iterations, "
                          f"rank(C) = {self.rk_C_[-1]}")
                break

        self._get_a_est(normed=normed)

    def pgd_l0(self, reg_l0=1, step="opt", n_it=10000, a_init=None,
               normed=False, save_iters=False, verbose=True):
        """
        Proximal gradient descent solver with rank penalty on C.

        Parameters
        ----------
        reg_l1 : rank of the matrix C
        step : gradient step
        n_it : max number of iterations
        a_init : gains at initialization
        normed : indicate if estimated gain vector should be normalized
        save_iters : keep all iterations
        verbose : helper message

        Parameters
        ----------
        reg_l0 : rank of the projected matrix C_
        step : gradient step
        n_it : max number of iterations
        a_init : gains at initialization
        normed : indicate if estimated gain vector should be normalized
        save_iters : keep all iterations
        verbose : helper message

        """
        if step == "opt":
            step = self.step_opt
        if a_init is None:
            self.a_init = np.ones((self.M, 1))
        else:
            self.a_init = a_init
        if save_iters is True:
            self.a_list = []

        self.C_ = self.a_init @ self.a_init.T.conj()
        self.rk_C_ = np.zeros((n_it))
        for n_ in range(n_it):
            C_old = self.C_
            self.Z_ = self.C_ + step/2*self.S*(self.Y - self.S*self.C_)
            self.C_, self.rk_C_[n_] = prox_ht((self.Z_ + self.Z_.T.conj())/2,
                                                   reg_l0)
            self._get_a_est(normed=normed)
            if save_iters is True:
                self.a_list.append(self.a_est)

            if np.linalg.norm(C_old - self.C_) < 1e-7:
                self.rk_C_ = self.rk_C_[0:n_+1]
                if verbose:
                    print(f"pgd-l0 done, {n_} iterations")
                break

        self._get_a_est(normed=normed)

    def vanilla(self, normed=False):
        """
        Vanilla apprach by element-wise division.

        Parameters
        ----------
        normed : indicate if estimated gain vector should be normalized

        """
        self.C_ = self.Y/self.S
        self._get_a_est(normed=normed)

    def scale_to(self, a_ref):
        """
        Align estimated gains onto a_ref in the least square sense.

        """
        alpha = (self.a_est.T.conj()@a_ref) / (self.a_est.T.conj()@self.a_est)
        self.a_est *= alpha
        try:
            for i, a_est in enumerate(self.a_tune):
                alpha = (a_est.T.conj()@a_ref) / (a_est.T.conj()@a_est)
                self.a_tune[i, :] = a_est * alpha
        except AttributeError:
            pass


def prox_st(mat, th):
    """
    Soft-thresholding operator on singular values of mat with threshold th.

    """
    u, s, vh = np.linalg.svd(mat)
    s = np.maximum(s - th, 0.0)
    s_card = (s != 0).sum()
    return u @ np.diag(s) @ vh, s_card


def prox_ht(mat, k):
    """
    Hard-thresholding operator on singular values of mat preserving the k
    largest non-zero values.

    """
    u, s, vh = np.linalg.svd(mat)
    s[k::] = 0
    return u @ np.diag(s) @ vh, k
