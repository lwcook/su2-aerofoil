from __future__ import division
import pdb
import pickle
import copy
import json
json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import utilities as utils
import base_aerofoil as base

def plot_surface(filenameIN, bTBox, **kwargs):

    Xlist, Ylist, surfacexy = base.readMeshFile(filename = filenameIN)

    surfacexyplot = np.zeros([surfacexy.shape[0]+1,2])
    surfacexyplot[0:-1,:] = surfacexy
    surfacexyplot[-1,:] = surfacexy[0,:]

    line, = plt.plot(surfacexyplot[:,0],surfacexyplot[:,1],**kwargs)

#    if bTBox:
#        dist, point1, point2, point3, point4 = check_box_fit(surfacexy)
#        plt.plot([point1[0], point2[0]], [point1[1], point2[1]],**kwargs)
#        plt.plot([point2[0], point4[0]], [point2[1], point4[1]],**kwargs)
#        plt.plot([point4[0], point3[0]], [point4[1], point3[1]],**kwargs)
#        plt.plot([point3[0], point1[0]], [point3[1], point1[1]],**kwargs)

    return line

def plot_mesh(filenameIN='case_mesh.su2', fig=None):

    Xlist, Ylist, surfacexy = base.readMeshFile(filename = filenameIN)
    if fig is None:
        fig, (ax1) = plt.subplots(1,1,sharex=True,figsize=(10,8))
    plt.figure(fig.number)

    line, = plt.plot(Xlist, Ylist, color=[0.6, 0.6, 0.6], alpha=0.2)
#    line, = plt.plot(Xlist, Ylist, color='k', alpha=1)

def plot_aerofoil(filenameIN='case_mesh.su2', fig=None, plotargs={}):

    Xlist, Ylist, surfacexy = base.readMeshFile(filename = filenameIN)
    if fig is None:
        fig, (ax1) = plt.subplots(1,1,sharex=True,figsize=(10,8))
    plt.figure(fig.number)

    foil = base.Aerofoil(reset=False)

    dist, p1, p2, p3, p4 = foil.boxFit(surfacexy)
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k')
    plt.plot([p3[0], p1[0]], [p3[1], p1[1]], 'k')
    plt.plot([p3[0], p4[0]], [p3[1], p4[1]], 'k')
    plt.plot([p2[0], p4[0]], [p2[1], p4[1]], 'k')

    N = int(len(surfacexy)/2)
    x1, y1 = surfacexy[0:N, 0], surfacexy[0:N, 1]
    x2, y2 = surfacexy[N:, 0], surfacexy[N:, 1]
    plt.plot(x1, y1, **plotargs)
    plt.plot(x2, y2, **plotargs)

    plt.fill_between(x1, 0, [y+0.000001 for y in y1], color=[1, 1, 1])
    plt.fill_between(x2, 0, [y-0.000001 for y in y2], color=[1, 1, 1])

#    return dist, p1, p2, p3, p4

def plot_surface_flow(filename='surface_flow.csv', fig=None):

    x_up, p_up, f_up, x_lo, p_lo, f_lo = base.readSurfaceFlow(filename)

    if fig is None:
        fig, (ax1) = plt.subplots(1,1,sharex=True,figsize=(15,10))
    plt.figure(fig.number)
    ax1 = plt.gca()

#    plt.scatter(x_up, p_up, c=utils.blue, label='Upper', lw=0, s=10)
#    plt.scatter(x_lo, p_lo, c=utils.red, label='Lower', lw=0, s=10)
    plt.plot(x_up, p_up, c=utils.blue, label='Upper')
    plt.plot(x_lo, p_lo, c=utils.red, label='Lower')

    ax2 = ax1.twinx()
    plt.sca(ax2)
    plt.plot(x_up, f_up, c=utils.blue, linestyle='dotted')
    plt.plot(x_lo, f_lo, c=utils.red, linestyle='dotted')
#    plt.plot([], [], c=utils.blue, linestyle='dashed', label='$C_f$ Upper')
#    plt.plot([], [], c=utils.red, linestyle='dashed', label='$C_f$ Lower')
    ax1.set_ylim([-1.0, 1.5])
    ax1.set_ylabel('$-C_p$')

    plt.sca(ax1)
    plt.xlabel('x/c')
    plt.legend(loc='upper right')
    plt.tight_layout()

def plot_surface_adjoint(filename='surface_flow.csv', fig=None):

    x_up, s_up, x_lo, s_lo = base.readSurfaceAdjoint(filename)

    if fig is None:
        fig, (ax1) = plt.subplots(1,1,sharex=True,figsize=(15,10))
    plt.figure(fig.number)
    ax1 = plt.gca()

    plt.plot(x_up, s_up, c=utils.blue, label='Upper')
    plt.plot(x_lo, s_lo, c=utils.red, label='Lower')
#    plt.plot([], [], c=utils.blue, linestyle='dashed', label='$C_f$ Upper')
#    plt.plot([], [], c=utils.red, linestyle='dashed', label='$C_f$ Lower')
#    ax1.set_ylim([-1.0, 1.5])
    ax1.set_ylabel('Surface Sensitivity')

    plt.sca(ax1)
    plt.xlabel('x/c')
    plt.legend(loc='upper right')
    plt.tight_layout()

def plot_field(filename='restart_flow.dat', fig=None, var='p'):

    X = np.zeros(100000)
    Y = np.zeros(100000)
    P = np.zeros(100000)
    T = np.zeros(100000)
    C = np.zeros(100000)
    M = np.zeros(100000)
    iNode1, iNode2,  = np.zeros(100000,int), np.zeros(100000,int)
    iNode3, iEleIndex = np.zeros(100000,int), np.zeros(100000,int)

    lines = [line.rstrip('\n') for line in open(filename)]

    iLine, iData, iMesh = 0, 0, 0
    k = np.zeros(10,int)

    for il, line in enumerate(lines):

        il0 = 100
        if il == 1 and line.split()[0].split('=')[0].upper() == 'VARIABLES':
            try:
                Pindex = line.split('""').index('Pressure')
                Tindex = line.split('""').index('Temperature')
                Mindex = line.split('""').index('Mach')
            except:
                Pindex = line.split(',').index('"Pressure"')
                Tindex = line.split(',').index('"Temperature"')
                Mindex = line.split(',').index('"Mach"')
            Xindex = 0
            Yindex = 1
            il0 = 4
        elif il == 0 and line.find('"Pressure"') != -1:
            Pindex = line.split().index('"Pressure"')
            Tindex = line.split().index('"Temperature"')
            Mindex = line.split().index('"Mach"')
            Xindex = line.split().index('"x"')
            Yindex = line.split().index('"y"')
            il0 = 1

        if il>=il0:       # data starts on 4th line
            data = [_ for _ in utils.float_gen(line.split())]
            if len(data) < 6:
                break # Reached connectivity information
            else:
                if abs(data[Xindex]) < 3 and abs(data[Yindex]) < 3:
                    X[iData] = data[Xindex]
                    Y[iData] = data[Yindex]
                    P[iData] = data[Pindex]
                    T[iData] = data[Tindex]
                    M[iData] = data[Mindex]
                    iData = iData+1

    X, Y, P = X[0:iData], Y[0:iData], P[0:iData]
    T, M = T[0:iData], [max(m, 0.200001) for m in M[0:iData]]

    N = 2000
    xi = np.concatenate((np.linspace(-200, -1.01, N/100),
                        np.linspace(-1, -0.101, N/10),
                        np.linspace(-0.1, 1.1, N),
                        np.linspace(1.101, 2, N/10),
                        np.linspace(2.01, 200, N/100)
                        ), axis=0)
    yi = np.concatenate((np.linspace(-200, -1.01, N/100),
                        np.linspace(-1, -0.101, N/10),
                        np.linspace(-0.1, 0.1, N),
                        np.linspace(0.101, 1, N/10),
                        np.linspace(1.01, 200, N/100)
                        ), axis=0)

    pi = griddata((X,Y), P, (xi[None,:],yi[:,None]), method='linear')
    piplot = copy.copy(pi)
    mi = griddata((X,Y), M, (xi[None,:],yi[:,None]), method='linear')
    miplot = copy.copy(mi)

    if fig is None:
        fig, (ax1) = plt.subplots(1,1,sharex=True,figsize=(15,10))
    plt.figure(fig.number)

    for iy, pix in enumerate(pi):
        for ix, p in enumerate(pix):
#            piplot[iy, ix] = max([min([p, 60000]), 20000])
            piplot[iy, ix] = p

    for iy, mix in enumerate(mi):
        for ix, m in enumerate(mix):
#            miplot[iy, ix] = max([min([m, 1.2]), 0.3])
            miplot[iy, ix] = m


    mirange = [m if not np.isnan(m) else 0 for m in miplot.flatten()]
    M_min = max(0.1, min(mirange))
    M_max = min(1.5, max(mirange))
    v = np.linspace(M_min, M_max, 80)
    if var.lower() == 'p':
        plt.contourf(xi, yi, piplot, 80, cmap='RdBu_r')
    elif var.lower() == 'm':
        plt.contourf(xi, yi, miplot, v, cmap='RdBu_r')

    v2 = np.linspace(M_min, M_max, 13)
    plt.colorbar(ticks=v2)

def plot_history(filename = 'history.dat', fig=None):

    iters = []
    CL = []
    CD = []
    CFL = []
    with open(filename, 'r') as f:
        iterindex = 0
        CLindex = 1
        CDindex = 2
        CFLindex = 20
        for il, line in enumerate(f.readlines()):

            # Data starts on 3rd line
            if il >= 3:
                iters.append(float(line.split(',')[iterindex]))
                CL.append(float(line.split(',')[CLindex]))
                CD.append(float(line.split(',')[CDindex]))

    if fig is None:
        fig, (ax1) = plt.subplots(1,1,sharex=True,figsize=(15,10))
    plt.figure(fig.number)
    ax = plt.gca()

    ax.set_xlabel('SU2 Iteration')

    ax.plot(iters, CL, 'b')
#    ax.set_ylabel('$C_L$')
    for tk in ax.get_yticklabels():
        tk.set_color('b')
#    ax.set_ylim([0.1, 0.9])

    ax2 = ax.twinx()
    ax2.plot(iters, CD, 'r')
#    ax2.set_ylabel('$C_D$')
    for tk in ax2.get_yticklabels():
        tk.set_color('r')

    return ax, ax2
#    ax2.set_ylim([0, 0.05])

def plot_adjoint(filename='restart_adj_cl.dat', fig=None, adjlim=500):

    X = np.zeros(100000)
    Y = np.zeros(100000)
    S = np.zeros(100000)
    iNode1, iNode2,  = np.zeros(100000,int), np.zeros(100000,int)
    iNode3, iEleIndex = np.zeros(100000,int), np.zeros(100000,int)

    lines = [line.rstrip('\n') for line in open(filename)]

    iLine, iData, iMesh = 0, 0, 0
    k = np.zeros(10,int)

    for il, line in enumerate(lines):

        il0 = 100
        if il == 1 and line.split()[0].split('=')[0].upper() == 'VARIABLES':
            try:
                Sindex = line.split('""').index('Conservative_1')
            except:
                Sindex = line.split(',').index('"Conservative_1"')
            Xindex = 0
            Yindex = 1
            il0 = 4
        elif il == 0 and line.find('"Conservative_1"') != -1:
            Sindex = line.split().index('"Conservative_1"')
            Xindex = 1
            Yindex = 2
            il0 = 1

        if il>=il0:       # data starts on 4th line
            data = [_ for _ in utils.float_gen(line.split())]
            if len(data) < 6:
                break # Reached connectivity information
            else:
                if abs(data[Xindex]) < 3 and abs(data[Yindex]) < 3:
                    X[iData] = data[Xindex]
                    Y[iData] = data[Yindex]
                    S[iData] = data[Sindex]
                    iData = iData+1

    X, Y, S = X[0:iData], Y[0:iData], S[0:iData]

    N = 2000
    xi = np.concatenate((np.linspace(-200, -1.01, N/100),
                        np.linspace(-1, -0.101, N/10),
                        np.linspace(-0.1, 1.1, N),
                        np.linspace(1.101, 2, N/10),
                        np.linspace(2.01, 200, N/100)
                        ), axis=0)
    yi = np.concatenate((np.linspace(-200, -1.01, N/100),
                        np.linspace(-1, -0.101, N/10),
                        np.linspace(-0.1, 0.1, N),
                        np.linspace(0.101, 1, N/10),
                        np.linspace(1.01, 200, N/100)
                        ), axis=0)

    pi = griddata((X,Y), S, (xi[None,:],yi[:,None]), method='linear')
    piplot = copy.copy(pi)

    if fig is None:
        fig, (ax1) = plt.subplots(1,1,sharex=True,figsize=(15,10))
    plt.figure(fig.number)

    for iy, pix in enumerate(pi):
        for ix, p in enumerate(pix):
            piplot[iy, ix] = max([min([p, adjlim]), -adjlim])
#            piplot[iy, ix] = p

    plt.contourf(xi, yi, piplot, 80, cmap='RdBu_r')

#    v2 = np.linspace(-1, 1, 20)
#    plt.colorbar(ticks=v2)
    plt.colorbar()

