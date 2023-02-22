# -*- coding: utf-8 -*-

# Copyright 2023 Charles Vanwynsberghe

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

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
        readings.

        Parameters
        ----------
        Y : Measured covariance matrix
        S : Modeled covariance matrix in the diffuse field.

        """
        self.Y = Y
        self.S = S
        self.M = self.Y.shape[0]
        self.step_opt = 2 / np.max(np.abs(self.S)**2)  # max stable step

    def _get_a_est(self, normed=False):
        """
        get final gain vector (first singular vector of A)

        """
        self.a_est, _, u_est = np.linalg.svd((self.Ak + self.Ak.T.conj())/2)
        self.a_est = self.a_est[:, 0]
        self.a_est *= _[0]**0.5
        self.a_est = self.a_est[:, None]
        if normed:
            self.a_est /= self.a_est.mean()

    def solve_relaxed(self, reg_l1=0.5, step=0.1,
                      n_it=10000, a_init=None,
                      normed=False, save_iters=False,
                      verbose=True):
        """
        pgd-l1 solver.

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

        self.Ak = self.a_init @ self.a_init.T.conj()
        self.s_st_card = np.zeros((n_it))
        for n_ in range(n_it):
            Ak_old = self.Ak
            self.Zk = self.Ak + step*(1/2)*self.S.conj()*(self.Y - self.S*self.Ak)
            self.Ak, self.s_st_card[n_] = prox_g((self.Zk + self.Zk.T.conj())/2,
                                                 step*reg_l1)

            self._get_a_est(normed=normed)
            if save_iters is True:
                self.a_list.append(self.a_est)

            if np.linalg.norm(Ak_old - self.Ak) < 1e-6:
                self.s_st_card = self.s_st_card[0:n_+1]
                if save_iters is True:
                    self.a_list = np.array(self.a_list).squeeze()
                if verbose:
                    print(f"PG trace  done, {n_} its, "
                          f"||A||_0 {self.s_st_card[-1]}-sparse")
                break

        self._get_a_est(normed=normed)

    def solve_l0(self, reg_l0=1, step=0.1,
                 n_it=10000, a_init=None,
                 normed=False, save_iters=False, verbose=True):
        """
        pgd-l0 solver.

        Parameters
        ----------
        reg_l1 : rank of the matrix A
        step : gradient step
        n_it : max number of iterations
        a_init : gains at initialization
        normed : indicate if estimated gain vector should be normalized
        save_iters : keep all iterations
        verbose : helper message

        Parameters
        ----------
        reg_l0 : rank of the projected matrix Ak
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

        self.Ak = self.a_init @ self.a_init.T.conj()
        self.s_st_card = np.zeros((n_it))
        for n_ in range(n_it):
            Ak_old = self.Ak
            self.Zk = self.Ak + step*(1/2)*self.S*(self.Y - self.S*self.Ak)
            self.Ak, self.s_st_card[n_] = prox_g_2((self.Zk + self.Zk.T.conj())/2,
                                                   reg_l0)
            self._get_a_est(normed=normed)
            if save_iters is True:
                self.a_list.append(self.a_est)

            if np.linalg.norm(Ak_old - self.Ak) < 1e-7:
                self.s_st_card = self.s_st_card[0:n_+1]
                if verbose:
                    print(f"P-l0 done, {n_} its")
                break

        self._get_a_est(normed=normed)

    def solve_svd(self, normed=False):
        """
        Vanilla apprach.

        Parameters
        ----------
        normed : indicate if estimated gain vector should be normalized

        """
        self.Ak = self.Y/self.S
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


def prox_st(vect, lbda):
    """
    Soft thresholding operator. Threshold is lambda.

    """
    vect_st = np.sign(vect) * np.maximum(np.abs(vect) - lbda, 0.0)
    return vect_st


def prox_ht(vect, k):
    """
    Hard thresholding operator. Sparsity degree is k.

    """
    vect_st = vect.copy()
    vect_st[k::] = 0
    return vect_st


def prox_g(X, lbda):
    """
    Proximal operator for g(X) as Shatten norm trace(X)

    """
    U, s, Vh = np.linalg.svd(X)
    s_st = prox_st(s, lbda)
    s_st_card = (s_st != 0).sum()
    return U @ np.diag(s_st) @ Vh, s_st_card


def prox_g_2(X, k):
    """
    Proximal operator for g(X) as l0 norm card(|X|*)

    """
    U, s, Vh = np.linalg.svd(X)
    s_st = prox_ht(s, k)
    return U @ np.diag(s_st) @ Vh, k


def randnc(*shape, random_state=None):
    """
    Return complex normal distribution following Nc(0, 1).

    Parameters
    ----------
    *shape : int sequence
        Output shape.

    random_state : numpy.random.RandomState
        Numpy RandomState generator. The default is None.

    Returns
    -------
    (*shape) numpy array
        array sampled with mean=0, std=1

    """
    if random_state.__class__ == np.random.mtrand.RandomState:
        out = (random_state.randn(*shape) +
               1j*random_state.randn(*shape)) / 2**0.5
    else:
        out = (np.random.randn(*shape) + 1j*np.random.randn(*shape)) / 2**0.5
    return out


def corr(u1, u2):
    """
    Normalised corrlation between u1 and u2

    """
    u1_ = u1.flatten()
    u2_ = u2.flatten()
    return u1_@u2_.conj() / (u1_@u1_.conj() * u2_@u2_.conj())**0.5
