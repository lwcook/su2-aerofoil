import numpy as np
import subprocess
import pdb
import time
import json
import os
import sys
import shutil

import parallel_computation as pc
import continuous_adjoint as ca

import base_aerofoil as base

class SU2aerofoil(base.Aerofoil):

    def __init__(self, aerofoil, flow, base_path=None, output_path=None, verbose=False,
            lift=None, log_name=None, bottom_fidelity=5, top_fidelity=10,
            CFD_params={}):

        super(SU2aerofoil, self).__init__(base_path, output_path, log_name,
                verbose)

        self.CLtarg = lift
        if flow.upper() == 'RANS':
            self.flow = 'RANS'
        elif flow.upper() == 'EULER':
            self.flow = 'EULER'
        else:
            raise ValueError('Unsupported flow type' + str(flow))

        valid_aerofoils = ['NACA0012_LF1', 'NACA0012_LF2', 'NACA0012_LF3',
                'NACA0012_HF', 'NACA0012_EULER', 'RAE2822', 'NACA64A010']
        aerofoilstr = ', '.join(valid_aerofoils)

        if aerofoil not in valid_aerofoils:
            raise ValueError('''Unsupported aerofoil: '''+str(aerofoil)+'.'+
                'Please choose from '+aerofoilstr)
        else:
            self.aerofoil = aerofoil.upper()

        self.CFD_params = CFD_params
        self.top_fidelity = top_fidelity
        self.bottom_fidelity = bottom_fidelity


    def _makeCaseDirectory(self, case):

        basepath = self.basepath
        outputpath = self.outputpath
        if not os.path.isdir(outputpath):
            os.mkdir(outputpath)

        typepath = outputpath+'/CFD'
        if not os.path.isdir(typepath):
            os.mkdir(typepath)

        casepath = typepath+'/'+case

        if not os.path.isdir(casepath):
            os.mkdir(casepath)

        self.casepath = casepath

#        foil = 'NACA0012_LF1'
#        foil = 'NACA0012_LF2'
#        foil = 'NACA0012_HF'
#        foil = 'RAE2822'
#        foil = 'NACA64A010'

        configname = casepath+'/case_config.cfg'
        meshname = casepath+'/case_mesh_original.su2'
        flowname = casepath+'/solution_flow.dat'
        clname = casepath+'/solution_adj_cl.dat'
        cdname = casepath+'/solution_adj_cd.dat'

        self.box_size = [0.5, 0.08]
        self.marker = 'AIRFOIL' # default

        if self.aerofoil == 'NACA0012_HF':
            shutil.copyfile(basepath+'/config/turb_nasa.cfg', configname)
            shutil.copyfile(basepath+'/geometry/naca0012rans_ff10.su2', meshname)

            shutil.copyfile(basepath+'/restarts/solution_flow_ff10.dat', flowname)
            shutil.copyfile(basepath+'/restarts/solution_adj_cd_ff10.dat', cdname)

        elif self.aerofoil == 'NACA64A010':
            shutil.copyfile(basepath+'/config/turb_NACA64A010.cfg', configname)
            shutil.copyfile(basepath+'/geometry/mesh_NACA64A010_turb.su2', meshname)

            shutil.copyfile(basepath+'/restarts/solution_adj_cl_n64A010.dat', clname)
            shutil.copyfile(basepath+'/restarts/solution_adj_cd_n64A010.dat', cdname)
            self.marker = 'airfoil'

        elif self.aerofoil == 'NACA0012_LF1':
            shutil.copyfile(basepath+'/config/turb_NACA0012_sa.cfg', configname)
            shutil.copyfile(basepath+'/geometry/n0012_113-33.su2', meshname)
            shutil.copyfile(basepath+'/restarts/solution_flow_n0012_113-33.dat',
                    flowname)
            shutil.copyfile(basepath+'/restarts/solution_adj_cl_n0012_113-33.dat',
                    clname)
            shutil.copyfile(basepath+'/restarts/solution_adj_cd_n0012_113-33.dat',
                    cdname)

        elif self.aerofoil == 'NACA0012_LF2':
            shutil.copyfile(basepath+'/config/turb_NACA0012_sa.cfg', configname)
            shutil.copyfile(basepath+'/geometry/n0012_225-65.su2', meshname)
            shutil.copyfile(basepath+'/restarts/solution_adj_cl_n0012_225-65.dat',
                    clname)
            shutil.copyfile(basepath+'/restarts/solution_adj_cd_n0012_225-65.dat',
                    cdname)

        elif self.aerofoil == 'NACA0012_LF3':
            shutil.copyfile(basepath+'/config/turb_NACA0012_sa.cfg', configname)
            shutil.copyfile(basepath+'/geometry/n0012_449-129.su2', meshname)
            shutil.copyfile(basepath+'/restarts/solution_adj_cl_n0012_449-129.dat',
                    clname)
            shutil.copyfile(basepath+'/restarts/solution_adj_cd_n0012_449-129.dat',
                    cdname)

        elif self.aerofoil == 'RAE2822':
            self.box_size = [0.5, 0.07]

            shutil.copyfile(basepath+'/geometry/mesh_RAE2822_turb.su2', meshname)
            shutil.copyfile(basepath+'/config/turb_SA_RAE2822.cfg', configname)
            shutil.copyfile(basepath+'/restarts/solution_RAE2822_flow.dat', flowname)
            shutil.copyfile(basepath+'/restarts/solution_RAE2822_adj_cd.dat', cdname)
            shutil.copyfile(basepath+'/restarts/solution_RAE2822_adj_cl.dat', clname)

        elif self.aerofoil == 'NACA0012_Euler':
            shutil.copyfile(basepath+'/geometry/mesh_NACA0012_inv.su2', meshname)
            shutil.copyfile(basepath+'/config/inv_NACA0012.cfg', configname)
            shutil.copyfile(basepath+'/config/solution_flow_n0012_euler.dat', flowname)
            shutil.copyfile(basepath+'/config/solution_adj_cl_n0012_euler.dat', clname)
            shutil.copyfile(basepath+'/config/solution_adj_cd_n0012_euler.dat', cdname)

        else:
            raise ValueError('Invalid Aerofoil')

        with open(meshname, 'r') as f:
            for l in f.readlines():
                vals = [v.strip('\n').strip('\r').strip(' ') for v in l.split('=')]
                if vals[0].upper() == 'MARKER_TAG' and vals[1].upper() == 'AIRFOIL':
                    self.marker = vals[1]
                    break

        return casepath

    def evalLift(self, x_IN, u, bParallel=True, bAdjoint=False):
        return self._evalAerofoil(x_IN, u, bParallel, bAdjoint, output='L')

    def evalDrag(self, x_IN, u, bParallel=True, bAdjoint=False):
        return self._evalAerofoil(x_IN, u, bParallel, bAdjoint, output='D')

    def evalLiftAndDrag(self, x_IN, u, bParallel=True, bAdjoint=False):
        return self._evalAerofoil(x_IN, u, bParallel, bAdjoint, output='LandD')

    def evalDragAtLift(self, x_IN, u, bParallel=True, bAdjoint=False):
        return self._evalAerofoil(x_IN, u, bParallel, bAdjoint, output='DatL')

    def evalLitToDrag(self, x_IN, u, bParallel=True, bAdjoint=False):
        return self._evalAerofoil(x_IN, u, bParallel, bAdjoint, output='LtoD')

    def _evalAerofoil(self, x_IN, u, bParallel, bAdjoint, output):

        output = output.upper()
        if output not in ['L', 'D', 'DATL', 'LTOD', 'LANDD']:
            raise ValueError('Unknown output: ' + str(output))

        ##############################################################
        # Establish directory structure
        ##############################################################

        # Scale x to be correct magnitude
        xscale = (1./1000.0)
        fscale = 100.0
        x = np.array(x_IN)*xscale
        M = 0.77 + 0.05*u

        case = 'M'+str(float(int(M*1000))/1000.)
        casepath = self._makeCaseDirectory(case)

        if os.path.isfile(casepath + '/AoA.txt'):
            with open(casepath + '/AoA.txt', 'r') as Afile:
                AoA = max([min([float(Afile.readline()), 10]), -10])
        else:
            AoA = 2

        ##############################################################
        # Update config file
        ##############################################################
        config_file = casepath+'/case_config.cfg'
        ##############################################################
        self._updateSU2File('PHYSICAL_PROBLEM', self.flow.upper(), config_file)

        self._updateSU2File('RESTART_SOL', 'NO', config_file)

        self._updateSU2File('MACH_NUMBER', str(M), config_file)
        self._updateSU2File('AOA', AoA, config_file)
        self._updateSU2File('AoA', AoA, config_file)

        self._updateSU2File('MESH_FILENAME', 'case_mesh_original.su2',
                config_file)
        self._updateSU2File('MESH_OUT_FILENAME', 'mesh_out.su2', config_file)
        self._updateSU2File('SOLUTION_FLOW_FILENAME', 'solution_flow.dat', config_file)
        self._updateSU2File('SOLUTION_ADJ_FILENAME', 'solution_adj.dat', config_file)

        self._updateSU2File('WRT_SOL_FREQ', 1000, config_file)
        self._updateSU2File('WRT_CON_FREQ', 50, config_file)

        max_iters = 5000
        self._updateSU2File('EXT_ITER', max_iters, config_file)

        ## Convergence Parameters
#        self._updateSU2File('CONV_CRITERIA', 'RESIDUAL', config_file)
#        self._updateSU2File('RESIDUAL_MINVAL', -7, config_file)
#        self._updateSU2File('RESIDUAL_REDUCTION', 9, config_file)
        self._updateSU2File('CONV_CRITERIA', 'CAUCHY', config_file)

        eps = 0.01
        self._updateSU2File('STARTCONV_ITER', 200, config_file)
        self._updateSU2File('CAUCHY_FUNC_FLOW', 'LIFT', config_file)
        self._updateSU2File('CAUCHY_ELEMS', 30, config_file)
        self._updateSU2File('CAUCHY_EPS', eps, config_file)

        ## CFD Paramteres
        self._updateSU2File('MGLEVEL', 0, config_file)
        self._updateSU2File('MGCYCLE', 'W_CYCLE', config_file)
        self._updateSU2File('AD_COEFF_FLOW', (0.2, 0.5, 0.05), config_file)
        self._updateSU2File('AD_COEFF_ADJFLOW', (0.1, 0.0, 0.01), config_file)
        self._updateSU2File('CONV_NUM_METHOD_ADJFLOW', 'JST', config_file)
        self._updateSU2File('SENS_REMOVE_SHARP', 'NO', config_file)
#        self._updateSU2File('CFL_REDUCTION_TURB', 1.0, config_file)
#        self._updateSU2File('TIME_DISCRE_FLOW', 'EULER_IMPLICIT', config_file)
#        self._updateSU2File('CONV_NUM_METHOD_FLOW', 'ROE', config_file)
        for key in self.CFD_params:
            self._updateSU2File(key.upper(), self.CFD_params[key], config_file)


        ##############################################################
        # Deform Mesh
        ##############################################################


        self._prepDeform(x, config_file, top_fidelity=self.top_fidelity,
                bot_fidelity=self.bottom_fidelity, marker=self.marker)

        os.chdir(self.casepath)
        subprocess.call(['cp', 'case_mesh_original.su2', 'case_mesh.su2'])
        subprocess.call(["SU2_DEF", config_file])
        subprocess.call(['cp', 'mesh_out.su2', 'case_mesh.su2'])
        subprocess.call(['rm', 'mesh_out.su2'])

        # Check thickness constraint
        Xlist, Ylist, surfacexy = base.readMeshFile('case_mesh.su2')
        fit_dist = self.boxFit(surfacexy, box_size=self.box_size)[0]

        ##############################################################
        # Run CFD
        ##############################################################
        # Make sure config file is ready for solve, not mesh deformation
        self._resetConfigFile(config_file, marker=self.marker)
        self._updateSU2File('MESH_FILENAME', 'case_mesh.su2', config_file)

        print('Mach: ', M, '    AoA: ', AoA)
        time.sleep(0.5)
        CL, CD = -1, 1
        geometry_penalty, converged_penalty, separation_penalty = 0.0, 0.0, 0.0

        if fit_dist <= 0.00:
            ##############################################################
            # Direct CFD to find coefficients
            ##############################################################
            self._updateSU2File('MATH_PROBLEM', 'DIRECT', config_file)

            def fLD(AoA):
                self._updateSU2File('AOA', AoA, config_file)
                self._updateSU2File('AoA', AoA, config_file)
#                if os.path.isfile('solution_flow.dat'):
#                    self._updateSU2File('RESTART_SOL', 'YES', config_file)
#                else:
#                    self._updateSU2File('RESTART_SOL', 'NO', config_file)
                self._updateSU2File('RESTART_SOL', 'NO', config_file)
                try:
                    self.runCFD(config_file, bParallel=True)
                    su2_CL, su2_CD, _, _ = self.readLD()
                    su2_conv = self._checkConvergence(CL, CD, max_iters)
                except:
                    su2_CL, su2_CD = -10.0, 10.0
                    su2_conv = False

                if su2_conv:
                    shutil.copyfile('restart_flow.dat', 'solution_flow.dat')
                    self._updateSU2File('EXT_ITER', 0, 'solution_flow.dat')
                return su2_CL, su2_CD, su2_conv

            if output == 'DATL':
                aoas, ls, ds, converged = self.innerLoop(
                                        AoA, fLD, self.CLtarg, eps, bAdjoint)
                CL, CD = ls[-1], ds[-1]
            else:
                CL, CD, converged = fLD(AoA)

            ##############################################################
            # Adjoints to find gradient
            ##############################################################

            if converged and bAdjoint:

                lgrad, dgrad = self.LDGradient(config_file, residual=-2.0, eps=eps)
                lgrad = [g*fscale*xscale for g in lgrad]
                dgrad = [g*fscale*xscale for g in dgrad]

                if output == 'DATL':
                    # Fit linear to find sensitivites w.r.t aoa
                    pcoeffsd = np.polyfit(aoas[-2:], ds[-2:], deg=1)
                    dslope = pcoeffsd[0]
                    pcoeffsl = np.polyfit(aoas[-2:], ls[-2:], deg=1)
                    lslope = pcoeffsl[0]

                    # Process gradients with scaling etc.
                    datlgrad = []
                    for dg, lg in zip(cdgrad, clgrad):
                        new = dg - (dslope/lslope)*lg
                        datlgrad.append(new)
                        datlgrad = [g*fscale*xscale for g in datlgrad]

            ##############################################################
            # Post processing
            ##############################################################
            if converged:

                if self.flow.upper() == 'RANS':
                    xu, pu, fu, xl, pl, fl = base.readSurfaceFlow()
                    separated, pen = self.evalSeparationPenalty(xu, fu, pu)
                    if separated:
                        separation_penlaty = 20.0*M*(1 + 1000*abs(pen)) - 10.0

            else:

                subprocess.call(['rm', 'restart_flow.dat'])
                subprocess.call(['rm', 'solution_flow.dat'])
                subprocess.call(['rm', 'AoA.txt'])

                CL = -10.0
                CD = 10.0
                lgrad = [1*np.random.rand() for _ in x_IN]
                dgrad = [1*np.random.rand() for _ in x_IN]
                datlgrad = [1*np.random.rand() for _ in x_IN]
                if self.CLtarg is not None:
                    converged_penalty = 60.0*M*(1 + abs(CL - self.CLtarg)) - 30.0
                else:
                    converged_penalty = 60.0*M*(1 + abs(CL)) - 30.0

        else:
            geometry_penalty = 40.0*M*(1 + 100*fit_dist) - 20.0
            CL = -20.0
            CD = 20.0
            lgrad = [1*np.random.rand() for _ in x_IN]
            dgrad = [1*np.random.rand() for _ in x_IN]
            datlgrad = [1*np.random.rand() for _ in x_IN]


        penalty = geometry_penalty + converged_penalty + separation_penalty
        os.chdir(self.workingpath)

        self.logCFD(M, CL, CD, geometry_penalty, separation_penalty,
                converged_penalty, surfacexy)
        L = float(fscale*(CL - penalty))
        D = float(fscale*(CD + penalty))
        LD = float(fscale*(CL/CD + penalty))

        pdb.set_trace()

        if output == 'L':
            if bAdjoint:
                return L, lgrad
            else:
                return L
        elif output == 'D':
            if bAdjoint:
                return D, dgrad
            else:
                return D
        elif output == 'LANDD':
            if bAdjoint:
                return L, D, lgrad, dgrad
            else:
                return L, D
        elif output == 'LTOD':
            if bAdjoint:
                qgrad = []
                for ii in len(np.arange(lgrad)):
                    qgrad.append( (CD*lgrad[ii] - CL*dgrad[ii])/CD**2 )
                return LD, qgrad
            else:
                return LD
        elif output == 'DATL':
            if bAdjoint:
                return D, datlgrad
            else:
                return D

    def LDGradient(self, config_file, residual=-2.0, eps=1e-3):

        adj_step = 0.001
        max_iters = 100000
        self._updateSU2File('EXT_ITER', max_iters, config_file)
        self._updateSU2File('STARTCONV_ITER', 500, config_file)
        if (os.path.isfile('solution_adj_cl.dat') and
                os.path.isfile('solution_adj_cd.dat')):
            self._updateSU2File('RESTART_SOL', 'YES', config_file)
            self._updateSU2File('EXT_ITER', 0, 'solution_adj_cl.dat')
            self._updateSU2File('EXT_ITER', 0, 'solution_adj_cd.dat')
        else:
            self._updateSU2File('RESTART_SOL', 'NO', config_file)

#        self._updateSU2File('RESTART_SOL', 'YES', config_file)

        self._updateSU2File('CONV_CRITERIA', 'CAUCHY', config_file)
        self._updateSU2File('CAUCHY_EPS', eps, config_file)
        self._updateSU2File('CAUCHY_FUNC_ADJFLOW', 'SENS_GEOMETRY',
                config_file)

#        self._updateSU2File('CONV_CRITERIA', 'RESIDUAL', config_file)
#        self._updateSU2File('RESIDUAL_MINVAL', residual, config_file)

        self._updateSU2File('MATH_PROBLEM',
                'CONTINUOUS_ADJOINT', config_file)
        self._updateSU2File('OBJECTIVE_FUNCTION', 'LIFT', config_file)
        self._updateSU2File('OPT_OBJECTIVE', 'LIFT', config_file)

        self._updateSU2File('CONV_FILENAME', 'history_cl', config_file)
        self._updateSU2File('SURFACE_ADJ_FILENAME', 'surface_adjoint_cl', config_file)
        ca.continuous_adjoint(config_file, 8, step=adj_step)


        self._updateSU2File('MATH_PROBLEM',
                'CONTINUOUS_ADJOINT', config_file)
        self._updateSU2File('OBJECTIVE_FUNCTION', 'DRAG', config_file)
        self._updateSU2File('OPT_OBJECTIVE', 'DRAG', config_file)

        self._updateSU2File('CONV_FILENAME', 'history_cd', config_file)
        self._updateSU2File('SURFACE_ADJ_FILENAME', 'surface_adjoint_cd', config_file)
        ca.continuous_adjoint(config_file, 8, step=adj_step)

        CL, CD, clgrad, cdgrad = self.readLD()

        return clgrad, cdgrad

    def innerLoop(self, AoA, fun, target, eps, bAdjoint):

        num = 1
        ds, ls, aoas, xs, ys, asort = [], [], [], [], [], []
        converged = True
        CL, CD = 0, 0

        with open(self.casepath + '/Inner_loop.txt', 'w') as f:
            pass

        while True:

            bFit = True
            if len(aoas) == 0:
                bFit = False
            elif len(aoas) == 1:
                bFit = False
                if abs(target-CL) > eps:
                    AoA = AoA + np.sign(target - CL)*0.5
                else:
                    if np.sign(target - CL) != 0:
                        AoA = AoA + np.sign(target - CL)*0.005
                    else:
                        AoA = AoA + 0.005

            if bFit:
                xs = np.array(aoas)
                ys = target - np.array(ls)
                asort = np.argsort(aoas)
                xsort, ysort = xs[asort], ys[asort]

                if np.sign(ysort[0]) == np.sign(ysort[-1]):
                    pcoeffs = np.polyfit(xs[-2:], ys[-2:], deg=1)
                    AoA = np.roots(pcoeffs)[0]
                else:
                    for ii in np.arange(len(xs)-1):
                        if np.sign(ysort[ii+1]) != np.sign(ysort[ii]):
                            aleft = xsort[ii]
                            aright = xsort[ii+1]
                            break

                    if len(aoas) % 4 == 0 or len(aoas) % 4 == 1:
                        p = np.polyfit(xs[-2:], ys[-2:], deg=1)
                        AoA = np.roots(p)[0]
                    elif len(aoas) % 4 == 3:
                        AoA = 0.5*(aleft + aright)
                    else:
                        p = np.polyfit(xsort[ii:ii+2], ysort[ii:ii+2], deg=1)
                        AoA = np.roots(p)[0]

            CL, CD, conv = fun(AoA)

            with open(self.casepath + '/Inner_loop.txt', 'a') as f:
                f.write('AoA: '+str(AoA)+'    CL: '+str(CL)+'    CD: '+
                        str(CD)+'\r\n')

            aoas.append(AoA)
            ds.append(CD)
            ls.append(CL)

            ## Sort by closeness to target
            delLs = np.array([abs(l - target) for l in ls])
            amin_inds = np.argsort(delLs)[::-1]
            aoas_out = np.array(aoas)[amin_inds]
            ls_out = np.array(ls)[amin_inds]
            ds_out = np.array(ds)[amin_inds]

            # Decide whether to break out of loop
            if len(aoas) > 1:

                if num > 20 or not conv:
                    converged = False
                    break
                else:
                    converged = True

                delA = abs(aoas_out[-1] - aoas_out[-2])
                if abs(ls_out[-1] - target) < eps:

                    bBreak = True
                    if bAdjoint and delA > 0.01:
                        bBreak = False

                    with open('AoA.txt', 'w') as Afile:
                        Afile.write(str(AoA))

                    if bBreak:
                        break

            num += 1

        return aoas_out, ls_out, ds_out, converged

    def evalSeparationPenalty(self, xu, fu, pu):
        '''Evaluates whether undesired trailing edge separation has occured
        from an upper surface skin friction distribution'''
        # tes - trailing edge separation
        # sfb - shock foot bubble

        ## Find trailing edge separation
        tes_x = []
        tes_f = []
        in_TE_sep = False
        for i, (x, f) in enumerate(zip(xu[::-1], fu[::-1])):
            if not in_TE_sep:
                if f < 0:
                    in_TE_sep = True
                    tes_x.append(x)
                    tes_f.append(f)
                if x < 0.96: ## Separation before 0.95 chord is not TES
                    break
            else:
                if f < 0:
                    tes_x.append(x)
                    tes_f.append(f)
                else:
                    break ## Separation ends when f goes positive
        tes_x = tes_x[::-1]
        tes_f = tes_f[::-1]

        ## Find shock foot separation bubble
        ## Shock detection: max. gradient of < -5 over previous 0.15c
        sfb_x = []
        sfb_f = []
        in_shock_foot = False
        for i, (x, f, p) in enumerate(zip(xu, fu, pu)):
            if not in_shock_foot:
                i_shock = 0
                for k in np.arange(i+1)[::-1]:
                    if xu[k] < x - 0.15:
                        i_shock = k
                        break
                shockx = xu[i_shock:i+1]
                shockp = pu[i_shock:i+1]
                max_grad = 0
                if len(shockx) > 1:
                    grads = [(shockp[j]-shockp[j-1])/(shockx[j]-shockx[j-1])
                                for j in np.arange(1, len(shockx))]
                    max_grad = min(grads)

                if f < 0 and max_grad < -5:
                    in_shock_foot = True
            else:
                if f > 0:
                    break

            if in_shock_foot:
                sfb_x.append(x)
                sfb_f.append(f)

        ## Check separation cases
        separated = False
        if len(sfb_x) > 0 and len(tes_x) > 0:
            if (min(sfb_x) == min(tes_x)
                    and max(sfb_x) == max(tes_x)):
                separated = True
            elif np.ptp(sfb_x) + np.ptp(tes_x) > 0.2:
                separated = True
        elif len(sfb_x) > 0 and np.ptp(sfb_x) > 0.2:
            separated = True
        elif len(tes_x) > 0 and min(tes_x) < 0.9:
            separated = True

        if separated:
            penalty_te = abs(np.trapz(tes_x, tes_f))
            penalty_sf = abs(np.trapz(sfb_x, sfb_f))
            penalty = penalty_te + penalty_sf
        else:
            penalty = 0

        return separated, penalty


if __name__ == "__main__":
    print("Use run_.py scripts for running cases")
