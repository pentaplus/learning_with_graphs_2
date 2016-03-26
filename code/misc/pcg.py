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
#    nargout = kwargs["nargout"] if kwargs else None
#    varargin = cellarray(args)
#    nargin = 7-[A,b,tol,maxit,M1,M2,x0].count(None)+len(args)

#    if (nargin < 2):
#        error(message(char('MATLAB:pcg:NotEnoughInputs')))
#    atype, afun, afcnstr=iterchk(A,nargout=3)
#    if strcmp(atype,char('matrix')):
#        m,n=size(A,nargout=2)
#        if (m != n):
#            error(message(char('MATLAB:pcg:NonSquareMatrix')))
#        if not isequal(size(b),cat(m,1)):
#            error(message(char('MATLAB:pcg:RSHsizeMatchCoeffMatrix'),m))
#    else:
#        m=size(b,1)
    eps = np.finfo(float).eps

    m = b.shape[0]
#        n=copy(m)
    n = m
#        if not iscolumn(b):
#            error(message(char('MATLAB:pcg:RSHnotColumn')))
#    if (nargin < 3) or isempty(tol):
#        tol=1e-06
#    warned=0
#    if tol <= eps:
#        warning(message(char('MATLAB:pcg:tooSmallTolerance')))
#        warned=1
#        tol=copy(eps)
#    else:
#        if tol >= 1:
#            warning(message(char('MATLAB:pcg:tooBigTolerance')))
#            warned=1
#            tol=1 - eps
#    if (nargin < 4) or isempty(maxit):
#        maxit=min(n,20)
#    n2b=norm(b)
    n2b = norm(b)
    if (n2b == 0):
        x = np.zeros((n, 1))
        flag = 0
        relres = 0
        iter_ = 0
        resvec = 0
#        if (nargout < 2):
#            itermsg(char('pcg'),tol,maxit,0,flag,iter_,NaN)
        return x, flag, relres, iter_, resvec
        
#    if ((nargin >= 5) and not isempty(M1)):
#        existM1=1
#        m1type,m1fun,m1fcnstr=iterchk(M1,nargout=3)
#        if strcmp(m1type,char('matrix')):
#            if not isequal(size(M1),cat(m,m)):
#                error(message(char('MATLAB:pcg:WrongPrecondSize'),m))
#    else:
#    existM1 = 0
#    m1type = 'matrix'
    
#    if ((nargin >= 6) and not isempty(M2)):
#        existM2=1
#        m2type,m2fun,m2fcnstr=iterchk(M2,nargout=3)
#        if strcmp(m2type,char('matrix')):
#            if not isequal(size(M2),cat(m,m)):
#                error(message(char('MATLAB:pcg:WrongPrecondSize'),m))
#    else:
#    existM2 = 0
#    m2type = 'matrix'
    
#    if ((nargin >= 7) and not isempty(x0)):
#        if not isequal(size(x0),cat(n,1)):
#            error(message(char('MATLAB:pcg:WrongInitGuessSize'),n))
#        else:
#            x=copy(x0)
#    else:
    x = np.zeros((n, 1))
#    if ((nargin > 7) and strcmp(atype,char('matrix')) and strcmp(m1type,char('matrix')) and strcmp(m2type,char('matrix'))):
#        error(message(char('MATLAB:pcg:TooManyInputs')))
    
    flag = 1
    xmin = x
    imin = 0
    tolb = tol * n2b
#    r= b - iterapp(char('mtimes'),afun,atype,afcnstr,x,varargin[:])
    r = b - afun(x)
#    normr=norm(r)
    normr = norm(r)
#    normr_act=copy(normr)
    normr_act = normr
    if normr <= tolb:
        flag = 0
        relres = normr / n2b
        iter_ = 0
        resvec = normr
#        if (nargout < 2):
#            itermsg(char('pcg'),tol,maxit,0,flag,iter_,relres)
        return x, flag, relres, iter_, resvec
        
#    resvec=zeros(maxit + 1,1)
    resvec = np.zeros((maxit + 1, 1))
#    resvec[1,:] = normr
    resvec[0,0] = normr
    normrmin = normr
    rho = 1
    stag = 0
    moresteps = 0
    maxmsteps = min([np.floor(n / 50), 5, n - maxit])
    maxstagsteps = 3
    
    for ii in xrange(1, maxit + 1):
#        if existM1:
#            y=iterapp(char('mldivide'),m1fun,m1type,m1fcnstr,r,varargin[:])
#            if not all(isfinite(y)):
#                flag=2
#                break
#        else:
        y = r
#        if existM2:
#            z=iterapp(char('mldivide'),m2fun,m2type,m2fcnstr,y,varargin[:])
#            if not all(isfinite(z)):
#                flag=2
#                break
#        else:
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
            
#        q=iterapp(char('mtimes'),afun,atype,afcnstr,p,varargin[:])
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
#        resvec[ii + 1, 1] = normr
        resvec[ii,0] = normr
        if normr <= tolb or stag >= maxstagsteps or moresteps:
            r = b - afun(x)
            normr_act = norm(r)
#            resvec[ii + 1,1]=normr_act
            resvec[ii,0] = normr_act
            if (normr_act <= tolb):
                flag = 0
                iter_ = ii
                break
            else:
                if stag >= maxstagsteps and moresteps == 0:
                    stag = 0
                moresteps += 1
                if moresteps >= maxmsteps:
#                    if not warned:
#                        warning(message(char('MATLAB:pcg:tooSmallTolerance')))
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
        r_comp=b - afun(xmin)
        if norm(r_comp) <= normr_act:
            x = xmin
            iter_ = imin
            relres = norm(r_comp) / n2b
        else:
            iter_ = ii
            relres = normr_act / n2b
    if flag <= 1 or flag == 3:
        resvec=resvec[0:ii,:]
    else:
        resvec=resvec[0:ii-1,:]
#    if (nargout < 2):
#        itermsg(char('pcg'),tol,maxit,ii,flag,iter_,relres)
    return x, flag, relres, iter_, resvec
