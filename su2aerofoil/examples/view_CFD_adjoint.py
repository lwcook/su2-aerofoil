import numpy as np
import os
import sys

import matplotlib.pyplot as plt

import aerofoil_plotting as setup

def main(case, desstr):

    workingdir = os.getcwd()
    os.chdir('output/' + case)

#    setup.plot_aerofoil('mesh_out.su2')
#    plt.xlim([-0.3, 1.3])
#    plt.ylim([-0.5, 0.5])
#    plt.show()

    fig, ((a1, a2, a3), (a4, a5, a6)) = plt.subplots(2, 3, figsize=(18, 12))

    if True:
        if os.path.isfile('restart_adj_cl.dat'):
            plt.sca(a1)
            setup.plot_adjoint('restart_adj_cl.dat', fig=fig, adjlim=50000)
            setup.plot_aerofoil('case_mesh.su2', fig=fig, plotargs={'c':'k'})
            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.7, 0.9])
            plt.gca().set_title('Lift Adjoint')
            plt.tight_layout()

    if True:
        if os.path.isfile('restart_adj_cd.dat'):
            plt.sca(a4)
            setup.plot_adjoint('restart_adj_cd.dat', fig=fig, adjlim=50000)
            setup.plot_aerofoil('case_mesh.su2', fig=fig, plotargs={'c':'k'})
            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.7, 0.9])
            plt.gca().set_title('Drag Adjoint')
            plt.tight_layout()

    plt.sca(a2)
    if os.path.isfile('surface_adjoint_cl.csv'):
        setup.plot_surface_adjoint('surface_adjoint_cl.csv', fig=fig)
        plt.gca().set_title('Surface Lift Sensitivity')

    plt.sca(a5)
    if os.path.isfile('surface_adjoint_cd.csv'):
        setup.plot_surface_adjoint('surface_adjoint_cd.csv', fig=fig)
        plt.gca().set_title('Surface Drag Sensitivity')

    plt.sca(a3)
    if os.path.isfile('history_cl_adjoint.dat'):
        ax1, ax2 = setup.plot_history('history_cl_adjoint.dat', fig=fig)
        ax1.set_label('Sens_Geo')
        ax2.set_label('Sens_Mach')
    elif os.path.isfile('history_adjoint.dat'):
        ax1, ax2 = setup.plot_history('history_adjoint.dat', fig=fig)
        ax1.set_label('Sens_Geo')
        ax2.set_label('Sens_Mach')
    plt.gca().set_title('Convergence')
    plt.tight_layout()

    plt.sca(a6)
    if os.path.isfile('history_cd_adjoint.dat'):
        ax1, ax2 = setup.plot_history('history_cd_adjoint.dat', fig=fig)
        ax1.set_label('Sens_Geo')
        ax2.set_label('Sens_Mach')
    plt.gca().set_title('Convergence')
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        case = str(sys.argv[1])
    else:
        case = 'M0.8'

    if len(sys.argv) > 2:
        desstr = '_' + str(sys.argv[2])
    else:
        desstr = '_MV'

    main('RansHM/CFD/'+case, desstr)
