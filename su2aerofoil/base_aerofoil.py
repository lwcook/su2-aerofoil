import numpy as np
import subprocess
import pdb
import time
import json
import os
import sys
import shutil

import mesh_deformation as md
import parallel_computation as pc
import continuous_adjoint as ca

def _floatGen(iterable):
    for obj in iterable:
        try:
            val = float(obj)
            yield val
        except TypeError:
            pass

class Aerofoil(object):

    def __init__(self, base_path=None, output_path=None, log_name=None,
            verbose=False, reset=True):

        self.verbose = verbose

        if log_name is None:
            self.log_name = ''
        else:
            self.log_name = log_name

        self.workingpath = os.getcwd()
        if base_path is None:
            self.basepath = self.workingpath + '/external/EulerAerofoil'
        else:
            self.basepath = self.workingpath + '/' + base_path.strip('/')
        if output_path is None:
            self.outputpath = self.workingpath + '/output/Euler'
        else:
            self.outputpath = self.workingpath + '/' + output_path.strip('/')

        if reset:
            self.CFD_log = self.outputpath+'/'+self.log_name+'_CFD_log.txt'
            self.geometry_log = self.outputpath+'/'+self.log_name+'_geometry_log.txt'
            with open(self.CFD_log, 'w') as f:
                pass ## reset logs
            with open(self.geometry_log, 'w') as f:
                pass ## reset logs

    def projectCFD(self, config_file):
        subprocess.call(['SU2_DOT', config_file])

    def runCFD(self, config_file, bParallel=True):
        if bParallel:
            pc.parallel_computation(config_file, 8)
        else:
            subprocess.call(['SU2_CFD', config_file])

    def readLD(self):
        cl, cl, clg, cdg = [], [], [], []
        with open('forces_breakdown.dat','r') as forces:
            for line in forces:
                if line.find('Total CL') != -1:
                    cl = float(line.split()[2])
                if line.find('Total CD') != -1:
                    cd = float(line.split()[2])
                    break
        if os.path.isfile('of_grad_cl.dat'):
            with open('of_grad_cl.dat', 'r') as f:
                for line in f:
                    if "GRADIENT" not in line:
                        clg.append(float(line.split(',')[1]))
        if os.path.isfile('of_grad_cd.dat'):
            with open('of_grad_cd.dat', 'r') as f:
                for line in f:
                    if "GRADIENT" not in line:
                        cdg.append(float(line.split(',')[1]))

        return cl, cd, clg, cdg

    def logCFD(self, M, CL, CD, geo_pen, sep_pen, conv_pen, geometry):

        ## Write CFD log for every CFD run
        with open(self.CFD_log, 'a') as f:
            f.write('Mach: ' + str(M) + '    CL : ' + str(CL) +
                    '    CD : ' + str(CD) +
                    '    Geometry Penalty: ' + str(geo_pen) +
                    '    Separation Penalty: ' + str(sep_pen) +
                    '    Convergence Penalty: ' + str(conv_pen) +
                    '\r\n')

        ## Write Geometry log every new design
        with open(self.geometry_log, 'r') as f:
            bWrite=False
            lines = f.readlines()
            if len(lines) > 0:
                last_geom = json.loads(lines[-1])
                if not np.array_equal(np.array(last_geom), np.array(geometry)):
                    bWrite=True
            else:
                bWrite = True

            if bWrite:
                with open(self.geometry_log, 'a') as f:
                    f.write(json.dumps(np.array(geometry).tolist()) + '\r\n')


    def evalAeroroil(self):
        raise ValueError('Need to override this in child class')

    def _prepDeform(self, x, config_file, top_fidelity=10, bot_fidelity=5,
            c_start=0.025, c_end=0.95, marker='airfoil'):

        # Defining the aerofoil parameterization scheme - 10 design variables
        x_locs = np.linspace(c_start, c_end, top_fidelity).tolist() +\
                    np.linspace(c_start, c_end, bot_fidelity).tolist()
        x_surface = [1]*top_fidelity + [0]*bot_fidelity

        kind_str = 'HICKS_HENNE'
        param_str = ' ' +str(x_surface[0])+ ', '  +str(x_locs[0])+  ' '
        marker_str = marker
        value_str = str(x[0])
        definition_str = '( 1, 1.0 | '+marker+' | ' +str(x_surface[0]) + \
            ' , '+str(x_locs[0])+'  )'
        for ii in range(1,len(x)):
            kind_str += ', HICKS_HENNE'
            param_str += '; ' +str(x_surface[ii])+ ', '  +str(x_locs[ii])+  ' '
            value_str += ', ' + str(x[ii])
#            marker_str += '; airfoil'
            definition_str += '; ( 1, 1.0 | '+marker+' | ' +str(x_surface[ii]) + \
                ' , '+str(x_locs[ii])+'  )'

        self._updateSU2File('DV_KIND', kind_str, config_file)
        self._updateSU2File('DV_PARAM', param_str, config_file)
        self._updateSU2File('DV_VALUE', value_str, config_file)
        self._updateSU2File('DV_MARKER', marker_str, config_file)
        self._updateSU2File('DEFINITION_DV', definition_str, config_file)
        self._updateSU2File('MARKER_DESIGNING', marker, config_file)

    def _updateSU2File(self, param, value, filein="case_config.cfg"):

        fileout = 'temp_config.cfg'

        bfoundparam = False
        with open(filein,'r') as cfile:
            with open(fileout,'wb') as cfileout:
                for line in cfile:
                    if str(line[0:len(param)]) == str(param) \
                            and str(line[len(param)]) == '=':
                        bfoundparam = True
                        cfileout.write(str(param) + '= ' + str(value) + '\n')
                    else:
                        cfileout.write(line)

        if bfoundparam == False:
            print 'Unable to find parameter to edit: ' + str(param)

        subprocess.call(["mv", fileout, filein])
        return filein

    def _commentSU2File(self, param, filein="case_config.cfg"):

        fileout = 'temp_config.cfg'

        bfoundparam = False
        with open(filein,'r') as cfile:
            with open(fileout,'wb') as cfileout:
                for line in cfile:
                    if str(line[0:len(param)]) == str(param) \
                            and str(line[len(param)]) == '=':
                        bfoundparam = True
                        cfileout.write('%' + line)
                    else:
                        cfileout.write(line)

        if bfoundparam == False:
            print 'Unable to find parameter to edit: ' + str(param)

        subprocess.call(["mv", fileout, filein])
        return filein

    def _resetConfigFile(self, config_file, marker='airfoil'):
        pass
        self._updateSU2File('DV_KIND', 'HICKS_HENNE', config_file)
        self._updateSU2File('DV_PARAM', '( 1, 0.5 )', config_file)
        self._updateSU2File('DV_VALUE', '0.00', config_file)
        self._updateSU2File('DV_MARKER', marker, config_file)

#        self._commentSU2File('DV_KIND', config_file)
#        self._commentSU2File('DV_PARAM', config_file)
#        self._commentSU2File('DV_VALUE', config_file)
#        self._commentSU2File('DV_MARKER', config_file)

    def _checkConvergence(self, CL, CD, max_iters, fname='history.dat'):
        ## Check convergence
        converged = True
        iters = []
        with open(fname, 'r') as f:
            iterindex = 0
            for il, line in enumerate(f.readlines()):
                # Data starts on 3rd line
                if il >= 3:
                    iters.append(float(line.split(',')[iterindex]))

        if iters[-1] >= max_iters-20:
            converged = False
        if abs(CL) > 100 or abs(CD) > 10:
            converged = False
        return converged

    def boxFit(self, surfacexy, box_size=[0.5, 0.08]):
        box_size = np.array(box_size)

        LE, TE, Nsurf = np.array([0.,0.]), surfacexy[-1], len(surfacexy)/2
        dists_upper = []
        dists_lower = []
        ## Connected to upper surface
        for point in surfacexy:

            if point[0] < 1.0 - box_size[0]:

                bdist = []
                boxB = [point[0] + box_size[0], point[1]]
                for other in surfacexy:
                    bdist.append(np.linalg.norm(np.array(other) - np.array(boxB)))
                pointB = surfacexy[np.argmin(bdist)]
                distB = max([0., boxB[1] - pointB[1], boxB[0] - pointB[0]])

                bdist = []
                boxC = [point[0], point[1] - box_size[1]]
                for other in surfacexy:
                    bdist.append(np.linalg.norm(np.array(other) - np.array(boxC)))
                pointC = surfacexy[np.argmin(bdist)]
                distC = max([0., pointC[1] - boxC[1]])

                bdist = []
                boxD = [point[0] + box_size[0], point[1] - box_size[1]]
                for other in surfacexy:
                    bdist.append(np.linalg.norm(np.array(other) - np.array(boxD)))
                pointD = surfacexy[np.argmin(bdist)]

                distD = max([0., pointD[1] - boxD[1], boxD[1] - pointB[1]])

                dists_upper.append(distB + distC + distD)

            else:
                dists_upper.append(1.)

        i_upper = np.argmin(dists_upper)
        dist_upper = dists_upper[i_upper]

        ## Connected to lower surface
        for point in surfacexy:

            if point[0] < 1.0 - box_size[0]:

                bdist = []
                boxB = [point[0] + box_size[0], point[1]]
                for other in surfacexy:
                    bdist.append(np.linalg.norm(np.array(other) - np.array(boxB)))
                pointB = surfacexy[np.argmin(bdist)]
                distB = max([0., pointB[1] - boxB[1], boxB[0] - pointB[0]])

                bdist = []
                boxC = [point[0], point[1] + box_size[1]]
                for other in surfacexy:
                    bdist.append(np.linalg.norm(np.array(other) - np.array(boxC)))
                pointC = surfacexy[np.argmin(bdist)]
                distC = max([0., boxC[1] - pointC[1]])

                bdist = []
                boxD = [point[0] + box_size[0], point[1] + box_size[1]]
                for other in surfacexy:
                    bdist.append(np.linalg.norm(np.array(other) - np.array(boxD)))
                pointD = surfacexy[np.argmin(bdist)]

                distD = max([0., boxD[1] - pointD[1], pointB[1] - boxD[1]])

                dists_lower.append(distB + distC + distD)

            else:
                dists_lower.append(1.)

        i_lower = np.argmin(dists_lower)
        dist_lower = dists_lower[i_lower]

        if dist_lower < dist_upper:
            dist = dist_lower
            point1 = np.array(surfacexy[i_lower])
            point2 = np.array([point1[0] + box_size[0], point1[1]])
            point3 = np.array([point1[0], point1[1] + box_size[1]])
            point4 = np.array([point1[0] + box_size[0], point1[1] + box_size[1]])
        else:
            dist = dist_upper
            point3 = np.array(surfacexy[i_upper])
            point4 = np.array([point3[0] + box_size[0], point3[1]])
            point1 = np.array([point3[0], point3[1] - box_size[1]])
            point2 = np.array([point3[0] + box_size[0], point3[1] - box_size[1]])

        return dist, point1, point2, point3, point4

def readMeshFile(filename = "mesh_NACA0012_inv.su2", filetype='su2'):

    lines = [line.rstrip('\n') for line in open(filename)]
    Nlines = len(lines)

    X = np.zeros(Nlines+1)
    Y = np.zeros(Nlines+1)
    NodeNum = np.zeros(Nlines+1)
    Elements = np.zeros((Nlines+1, 10), int)
    surfacexy = None


    iData, iMesh, iSurf = 0, 0, 0

    for il, line in enumerate(lines):
        if line[0] != '%': # ignore % as comments
            if line.find('NDIME') != -1:
                case = 'dimension'
                NDIME = int(line[line.find('=')+1:])
            elif line.find('NELEM') != -1:
                case = 'connectivity'
                NELEM = int(line[line.find('=')+1:])
            elif line.find("NPOIN") != -1:
                case = 'nodes'
                NPOIN = int(line[line.find('=')+1:])
            elif line.find('NMARK') != -1:
                case = 'markers'
            elif line.lower().find('airfoil') != -1:
                case = 'airfoil'
            elif line.lower().find('farfield') != -1:
                case = 'farfield'
            else:
                if case == 'connectivity':
                    vals = [int(_) for _ in _floatGen(line.split())]
                    Elements[iMesh, 0:len(vals)] = vals
                    iMesh = iMesh + 1
                elif case == 'nodes':
                    vals = [val for val in _floatGen(line.split())]
                    X[iData], Y[iData] = vals[0], vals[1]
                    NodeNum[iData] = line.split()[-1]
                    iData = iData + 1
                elif case =='airfoil':
                    if line.find("MARKER_ELEMS") != -1:
                        surfacexy = np.zeros([int(line.split('=')[1]),2])
                    elif line.split()[0] == "3":
                        Node = int(line.split()[1])
                        surfacexy[iSurf,:] = [X[Node], Y[Node]]
                        iSurf += 1


    X, Y, NodeNum = X[0:iData], Y[0:iData], NodeNum[0:iData]
    Elements = Elements[0:iMesh,:]

    Xlist, Ylist = [], []

    # Elements holds data: eletype, node1, node2, ..., elementno.
    for iEle in range(0,(np.shape(Elements)[0])):

        Xlist.extend([X[Elements[iEle,1]],X[Elements[iEle,2]]])
        Xlist.append(None)
        Ylist.extend([Y[Elements[iEle,1]],Y[Elements[iEle,2]]])
        Ylist.append(None)

        Xlist.extend([X[Elements[iEle,2]],X[Elements[iEle,3]]])
        Xlist.append(None)
        Ylist.extend([Y[Elements[iEle,2]],Y[Elements[iEle,3]]])
        Ylist.append(None)

    return Xlist, Ylist, surfacexy

def readSurfaceFlow(filename='surface_flow.csv'):

    with open(filename, 'r') as f:
        lines = f.readlines()

    def getvals(line):
        return [s.strip('\n').strip(' ').strip("'").strip('"')
                for s in line.split(',')]

    entries = getvals(lines[0])

    i_upper, i_lower = [], []
    p_upper, p_lower = [], []
    f_upper, f_lower = [], []
    x_upper, x_lower = [], []
    for line in lines[1:]:
        point = {}
        for i, e in enumerate(entries):
            point[e] = float(getvals(line)[i])
        if point['x_coord'] < 0.9999999999 and point['x_coord'] > 0.000000001:
            if point['y_coord'] > 0:
                x_upper.append(point['x_coord'])
                p_upper.append(-1*point['Pressure_Coefficient'])
                f_upper.append(point['Skin_Friction_Coefficient_X'])
                i_upper.append(point['Global_Index'])
            else:
                x_lower.append(point['x_coord'])
                p_lower.append(-1*point['Pressure_Coefficient'])
                f_lower.append(point['Skin_Friction_Coefficient_X'])
                i_lower.append(point['Global_Index'])

    iusort = np.argsort(i_upper)
    x_upper = np.array(x_upper)[iusort]
    p_upper = np.array(p_upper)[iusort]
    f_upper = np.array(f_upper)[iusort]

    ilsort = np.argsort(i_lower)
    x_lower = np.array(x_lower)[ilsort]
    p_lower = np.array(p_lower)[ilsort]
    f_lower = np.array(f_lower)[ilsort]

    return x_upper, p_upper, f_upper, x_lower, p_lower, f_lower

def readSurfaceAdjoint(filename='surface_adjoint.csv'):

    with open(filename, 'r') as f:
        lines = f.readlines()

    def getvals(line):
        return [s.strip('\n').strip(' ').strip("'").strip('"')
                for s in line.split(',')]

    entries = getvals(lines[1])

    i_upper, i_lower = [], []
    s_upper, s_lower = [], []
    x_upper, x_lower = [], []
    for line in lines[2:]:
        point = {}
        for i, e in enumerate(entries):
            point[e] = float(getvals(line)[i])
        if point['x_coord'] < 0.9999999999 and point['x_coord'] > 0.000000001:
            if point['y_coord'] > 0:
                x_upper.append(point['x_coord'])
                s_upper.append(point['Sensitivity'])
                i_upper.append(point['Point'])
            else:
                x_lower.append(point['x_coord'])
                s_lower.append(point['Sensitivity'])
                i_lower.append(point['Point'])

    iusort = np.argsort(i_upper)
    x_upper = np.array(x_upper)[iusort]
    s_upper = np.array(s_upper)[iusort]

    ilsort = np.argsort(i_lower)
    x_lower = np.array(x_lower)[ilsort]
    s_lower = np.array(s_lower)[ilsort]

    return x_upper, s_upper, x_lower, s_lower

def readHistory(filename = 'history.dat'):

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
                CFL.append(float(line.split(',')[CFLindex]))

    return iters, CL, CD, CFL

if __name__ == "__main__":
    print("Use run_.py scripts for running cases")
