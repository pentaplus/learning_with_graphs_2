"""
Preconditioned Conjugate Gradients Method.

This module provides the function pcg, which is a shortened translation
of MATLAB's function pcg, since it only features the signatures
pcg(A, b), pcg(A, b, tol) and pcg(A, b, tol, maxit).
See the documentation for MATLAB's function pcg:
http://de.mathworks.com/help/matlab/ref/pcg.html
"""

from __future__ import division

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__date__ = "2016-02-28"

import numpy as np

from numpy.linalg import norm


def pcg(afun, b, tol = 1e-6, maxit = 20):
    eps = np.finfo(float).eps

    m = b.shape[0]
    n = m

    n2b = norm(b)
    if (n2b == 0):
        x = np.zeros((n, 1))
        flag = 0
        relres = 0
        iter_ = 0
        resvec = 0

        return x, flag, relres, iter_, resvec
        

    x = np.zeros((n, 1))
    
    flag = 1
    xmin = x
    imin = 0
    tolb = tol * n2b
    r = b - afun(x)
    normr = norm(r)
    normr_act = normr
    
    if normr <= tolb:
        flag = 0
        relres = normr / n2b
        iter_ = 0
        resvec = normr
        
        return x, flag, relres, iter_, resvec
        
    resvec = np.zeros((maxit + 1, 1))
    resvec[0, 0] = normr
    normrmin = normr
    rho = 1
    stag = 0
    moresteps = 0
    maxmsteps = min([np.floor(n / 50), 5, n - maxit])
    maxstagsteps = 3
    
    for ii in xrange(1, maxit + 1):
        y = r
        z = y
        rho1 = rho
        rho = float(r.T.dot(z))
        if rho == 0 or np.isinf(rho):
            flag = 4
            break
        if ii == 1:
            p = z
        else:
            beta = rho / rho1
            if beta == 0 or np.isinf(beta):
                flag = 4
                break
            p = z + beta * p
            
        q = afun(p)
        pq = float(p.T.dot(q))
        if pq <= 0 or np.isinf(pq):
            flag = 4
            break
        else:
            alpha = rho / pq
        if np.isinf(alpha):
            flag = 4
            break
        
        if norm(p) * abs(alpha) < eps * norm(x):
            stag += 1
        else:
            stag = 0
            
        x = x + alpha * p
        r = r - alpha * q
        normr = norm(r)
        normr_act = normr
        resvec[ii, 0] = normr
        if normr <= tolb or stag >= maxstagsteps or moresteps:
            r = b - afun(x)
            normr_act = norm(r)
            resvec[ii, 0] = normr_act
            if (normr_act <= tolb):
                flag = 0
                iter_ = ii
                break
            else:
                if stag >= maxstagsteps and moresteps == 0:
                    stag = 0
                moresteps += 1
                if moresteps >= maxmsteps:
                    flag = 3
                    iter_ = ii
                    break
                
        if normr_act < normrmin:
            normrmin = normr_act
            xmin = x
            imin = ii
        if stag >= maxstagsteps:
            flag = 3
            break
        
        
    if flag == 0:
        relres = normr_act / n2b
    else:
        r_comp = b - afun(xmin)
        if norm(r_comp) <= normr_act:
            x = xmin
            iter_ = imin
            relres = norm(r_comp) / n2b
        else:
            iter_ = ii
            relres = normr_act / n2b
            
    if flag <= 1 or flag == 3:
        resvec = resvec[:ii, :]
    else:
        resvec = resvec[:ii - 1, :]
        
    return x, flag, relres, iter_, resvec
    
