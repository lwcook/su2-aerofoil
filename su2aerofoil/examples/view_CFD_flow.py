import numpy as np
import os
import sys

import matplotlib.pyplot as plt

import src.aerofoil_plotting as setup
import src.utilities as utils

def main(case):

    workingdir = os.getcwd()
    os.chdir('output/' + case)

#    setup.plot_aerofoil('mesh_out.su2')
#    plt.xlim([-0.3, 1.3])
#    plt.ylim([-0.5, 0.5])
#    plt.show()

    utils.mpl2tex()
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 6))

    plt.sca(a1)
    if os.path.isfile('restart_flow.dat'):
        setup.plot_field('restart_flow.dat', fig=fig, var='m')
        setup.plot_mesh(fig=fig)
    elif os.path.isfile('solution_flow.dat'):
        setup.plot_field('solution_flow.dat', fig=fig, var='m')
        setup.plot_mesh(fig=fig)
    setup.plot_aerofoil('case_mesh.su2', fig=fig, plotargs={'c':'k'})
    plt.xlim([-0.3, 1.3])
    plt.ylim([-0.7, 0.9])
    plt.gca().set_title('Flow')
#    plt.axis('equal')
    plt.tight_layout()

    plt.sca(a2)
    if os.path.isfile('surface_flow.csv'):
        setup.plot_surface_flow('surface_flow.csv', fig=fig)

    plt.sca(a3)
    if os.path.isfile('history.dat') or os.path.isfile('history_direct.dat'):
        if os.path.isfile('history.dat'):
            ax1, ax2 = setup.plot_history('history.dat', fig=fig)
        elif os.path.isfile('history_direct.dat'):
            ax2, ax2 = setup.plot_history('history_direct.dat', fig=fig)
        ax1.set_ylabel('$C_L$')
        ax1.set_ylabel('$C_D$')
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        case = str(sys.argv[1])
    else:
        case = 'M0.8'

    if len(sys.argv) > 2:
        outdir = str(sys.argv[2])
    else:
        outdir = 'RansHM'

    main(outdir+'/CFD/'+case)
