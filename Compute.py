import numpy as np
import sys
import glob
import os
import re
import subprocess
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.lib.function_base import select
from scipy import stats

import Data
import colormaps

try:
    plt.rc('font', size=13, family="Times New Roman")
    plt.rcParams["text.usetex"] = True
except:
    print("no tex")

class Dumpobj():

    def __init__(self, path=None, dirname_pattern='equil_{}', dir_start=0, dir_end=0, fname_pattern='dump*.{}', Nevery=10000, Nrepeat=1, Nfreq=10000, end=10000, timestep=0.006, comm=None):

        if path == None:
            print('path was not given and set to current working directory.')
            path = os.getcwd()

        if dir_start > dir_end:
            print('dir_start cannot be larger than dir_end. dir_start is set to 0.')
            dir_start = 0

        if Nevery > Nfreq:
            print('Nevery cannot be larger than Nfreq. Nevery is set to Nfreq.')
            Nevery = Nfreq

        if Nevery > end:
            print('Nevery cannot be larger than end. Nevery is set to end.')
            Nevery = end

        if Nfreq > end:
            print('Nfreq cannot larger than end. Nfreq is set to end.')
            Nfreq = end

        self.path = path
        self.dirname_pattern = dirname_pattern
        self.dir_start = dir_start
        self.dir_end = dir_end
        self.fname_pattern = fname_pattern
        self.Nevery = Nevery
        self.Nrepeat = Nrepeat
        self.Nfreq = Nfreq
        self.end = end
        self.comm = comm

        self.timestep = timestep
        self.results = {
                'timestep': timestep
                }

        self.source_dirs = []
        for iind in range(dir_start, dir_end + 1):
            self.source_dirs.append(dirname_pattern.format(iind))

        self.freqs = np.linspace(Nfreq, end, int(end/Nfreq), dtype=int)

        fnames = []
        if path is not None: 
            fnames.append(glob.glob(os.path.join(path, self.source_dirs[0], self.fname_pattern.format(0)))[0])
        else:
            path = os.getcwd()
            fnames.append(glob.glob(os.path.join(path, self.source_dirs[0], self.fname_pattern.format(0)))[0])
        buff_path = path

        steps = [0]
        for dir_ind, source_dir in enumerate(self.source_dirs):

            ##
            # This part is being used temporarily due to running short on disk space
            #if dir_ind >= 20 and dir_ind < 30:
            #    path = os.path.join('/data/N2022_f2550', os.path.basename(path))
            #else:
            #    path = buff_path
            ##

            for freq_ind, freq in enumerate(self.freqs):
                patterns = [fname_pattern.format(i) for i in np.linspace(freq - self.Nevery*(self.Nrepeat -1), freq, self.Nrepeat, dtype=int)]

                for pattern_ind, pattern in enumerate(patterns):
                    if path is not None:
                        fnames.append(glob.glob(os.path.join(path, source_dir, pattern))[0])
                    else:
                        fnames.append(glob.glob(os.path.join(source_dir, pattern))[0])
                
                steps.append((dir_start + dir_ind)*end + freq)
        
        self.fnames = fnames
        self.steps = np.asarray(steps)
        self.results.update({'steps': np.asarray(steps)})

        rank = self.comm.Get_rank()

        if rank == 0:
            print("The number of files to be processed = {}".format(len(fnames)))

            f = open(self.fnames[0], 'r')
            box = []
            for i in range(5):
                f.readline()
            for i in range(3):
                line = f.readline().split()
                box.append([float(line[0]), float(line[1])])
            f.close()

            lx = box[0][1] - box[0][0]
            ly = box[1][1] - box[1][0]

            trj = np.loadtxt(self.fnames[-1], skiprows=9)
            trj = trj[trj[:,2] <= np.unique(trj[:,2])[:-2][-1]]
            bins = np.linspace(0, 80, 161)
            hist, bin_edges = np.histogram(trj[:,5], bins=bins)
            z = bin_edges[1:] - (bin_edges[1] - bin_edges[0])/2
            rho = hist/(lx * ly * (bin_edges[1] - bin_edges[0]))

            import lmfit

            def sigmoid(v, x): 
                return v['c']/(1.0 + np.exp(-v['a']*(x - v['b'])))

            def func2minimize(pars, x, y): 
                v = pars.valuesdict()
                m = sigmoid(v, x)

                return m - y 

            pars = lmfit.Parameters()
            pars.add_many(('a', -0.5), ('b', np.average(z)), ('c', 0.8))
            mi = lmfit.minimize(func2minimize, pars, args=(z, rho))
            popt = mi.params
            h = mi.params['b'] 

            self.box = box
            self.lx = lx
            self.ly = ly
            self.h = h

        else:
            self.box = None
            self.lx = None
            self.ly = None

        self.box = comm.bcast(self.box, root=0)
        self.lx = comm.bcast(self.lx, root=0)
        self.ly = comm.bcast(self.ly, root=0)

    def save_results(self):
        if self.comm.rank == 0:
            np.save('results.npy', self.results)

    def specify_timesteps(self, timesteps):
        '''
        Specify timesteps by providing an array
        '''

        self.Nrepeat = 1
        fnames = []
        for dir_ind, source_dir in enumerate(self.source_dirs):

            ##
            # This part is being used temporarily due to running short on disk space
            #if dir_ind >= 20:
            #    path = os.path.join('/data/N2022_f2550', os.path.basename(path))
            ##

            patterns = [self.fname_pattern.format(int(timestep)) for timestep in timesteps]

            for pattern_ind, pattern in enumerate(patterns):
                if self.path is not None:
                    fnames.append(glob.glob(os.path.join(self.path, source_dir, pattern))[0])
                else:
                    fnames.append(glob.glob(os.path.join(source_dir, pattern))[0])
        
        self.fnames = fnames
        print("The number of files to be processed = {}".format(len(fnames)))

    def run(self, computes, comm=None):
        # TODO: run several functions on the trj at the same time
        fnames = []

    def computeDensity(self, zlo, zhi, n_bins):
        self.zlo = zlo
        self.zhi = zhi
        self.n_bins = n_bins
        self.bins = np.linspace(zlo, zhi, n_bins)

        size = self.comm.Get_size()
        rank = self.comm.Get_rank()

        avg_rows_per_process = int(len(self.fnames)/size)

        start_row = rank * avg_rows_per_process
        end_row = start_row + avg_rows_per_process
        if rank == size - 1:
            end_row = len(self.fnames)

        density_tmp = np.empty([len(self.fnames), self.n_bins - 1])
        for iind in range(start_row, end_row):

            try:

                trj = np.loadtxt(self.fnames[iind], skiprows=9)

                f = open(self.fnames[iind],'r')
                for i in range(3):
                    f.readline()
                n_atoms = int(f.readline().split()[0])
                f.close()

                flag = 0
                shift = 1
                while (flag == 0):
                    if trj.shape[0] != n_atoms:
                        trj = np.loadtxt(self.fnames[iind - shift], skiprows=9)

                        f = open(self.fnames[iind - shift],'r')
                        for i in range(3):
                            f.readline()
                        n_atoms = int(f.readline().split()[0])
                        f.close()

                        shift +=1 
                        
                    else:
                        flag = 1

            except:
                'empty dump files can be created due to storage limit'

                flag = 0
                shift = 1
                while (flag == 0):

                    try:
                        trj = np.loadtxt(self.fnames[iind - shift], skiprows=9)

                        f = open(self.fnames[iind - shift],'r')
                        for i in range(3):
                            f.readline()
                        n_atoms = int(f.readline().split()[0])
                        f.close()

                        if trj.shape[0] == n_atoms:
                            flag = 1
                        else:
                            shift += 1

                    except:
                        shift += 1

            density_tmp[iind, :] = self._computeDensity(trj)

            del trj

        if rank == 0:

            for iind in range(1, size):
                start_row = iind*avg_rows_per_process
                end_row = start_row + avg_rows_per_process
                if iind == size - 1:
                    end_row = len(self.fnames)

                recv = np.empty([len(self.fnames), self.n_bins - 1])
                req = self.comm.Irecv(recv, source=iind)
                req.Wait()

                density_tmp[start_row:end_row] = recv[start_row:end_row]
        else:
            send = density_tmp
            req = self.comm.Isend(send, dest=0)
            req.Wait()

        if rank == 0:
            
            rho = Data.Data_TimeSerie2D(density_tmp, self.Nrepeat)

            density_final = rho.mean
            density_final = density_final.transpose()
                        
            res = Data.Data2D(z=density_final)
            xticks = np.linspace(0,len(self.source_dirs)*len(self.freqs),4) 
            plot_args = {
                'zmin': 0,
                'zmax': 0.8,
                'cmap': colormaps.cmaps['W2B_8'],
                'cbarticks': [0, 0.8],
                'cbarlabel': r'density ($m/\sigma^{3}$)',
                'xticks': xticks,
                'xticklabels': [format(i*self.Nfreq*0.006*pow(10,-6), '.4f') for i in xticks],
                'xlabel': r'time ($\times 10^{6} \tau$)',
                'yticks': np.linspace(0, self.n_bins-1, 5),
                'yticklabels': np.linspace(self.bins[0], self.bins[-1], 5),
                'ylabel': r'$z$ ($\sigma$)',
            }

            res.plot(save='density_evol.png', show=False, plot_args=plot_args)

            '''
            cmaps = {}
            from matplotlib.colors import ListedColormap
            #W2B = np.dstack((np.linspace(1,0,256), np.linspace(1,0,256), np.linspace(1,0,256)))
            W2B = np.dstack((np.linspace(1,0,8), np.linspace(1,0,8), np.linspace(1,0,8)))
            cmaps['W2B'] = ListedColormap(W2B[0], name='W2B')

            fig, ax = plt.subplots(figsize=(4, 3))
            #plt.rc('font',size=9)

            im = ax.imshow(density_final, vmin=0, vmax=0.8, cmap=cmaps['W2B'], origin='lower')
            ax.set_aspect('auto')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_ticks([0, 0.8])
            cbar.set_label(r'density ($m/\sigma^{3}$)')
            xticks = np.linspace(0,len(self.source_dirs)*len(self.freqs),4)
            ax.set_xticks(xticks)
            ax.set_xticklabels([format(i*self.Nfreq*0.006*pow(10,-6), '.4f') for i in xticks])
            ax.set_xlabel(r'time ($\times 10^{6} \tau$)')
            ax.set_yticks(np.linspace(0, self.n_bins-1,5))
            ax.set_ylim(0 - 0.5, self.n_bins-1 + 0.5)
            ax.set_yticklabels(np.linspace(self.bins[0], self.bins[-1], 5))
            ax.set_ylabel(r'$z$ ($\sigma$)')
            plt.tight_layout(pad=1,h_pad=None,w_pad=None,rect=None)
            plt.savefig("density_evol.png", dpi=1000)
            '''

            return density_final

    def _computeDensity(self, trj, **args):
        #trj_all = trj[trj[:,2] <= 4]
        trj_phase1 = trj[(trj[:,2] == 1) | (trj[:,2] == 3)]
        #trj_phase2 = trj[(trj[:,2] == 2) | (trj[:,2] == 4)]

        hist, bin_edges = np.histogram(trj_phase1[:,5], bins=self.bins)
        density_1 = hist/(self.lx*self.ly*(bin_edges[1]-bin_edges[0]))
        #hist, bin_edges = np.histogram(trj_phase2[:,5], bins=self.bins)
        #density_2 = hist/(self.lx*self.ly*(bin_edges[1]-bin_edges[0]))

        #del trj_all
        del trj_phase1
        #del trj_phase2

        density = density_1 #/ (density_1 + density_2) 
        
        return density

    def computeLocalfC(self, lx, ly, zlo, zhi, n_bins):
        self.lx = lx
        self.ly = ly
        self.zlo = zlo
        self.zhi = zhi
        self.n_bins = n_bins
        self.bins = np.linspace(zlo, zhi, n_bins)

        size = self.comm.Get_size()
        rank = self.comm.Get_rank()

        avg_rows_per_process = int(len(self.fnames)/size)

        start_row = rank * avg_rows_per_process
        end_row = start_row + avg_rows_per_process
        if rank == size - 1:
            end_row = len(self.fnames)

        density_tmp = np.empty([len(self.fnames), self.n_bins - 1])
        for iind in range(start_row, end_row):

            try:

                trj = np.loadtxt(self.fnames[iind], skiprows=9)

                f = open(self.fnames[iind],'r')
                for i in range(3):
                    f.readline()
                n_atoms = int(f.readline().split()[0])
                f.close()

                #print(n_atoms)

                flag = 0
                shift = 1
                while (flag == 0):
                    if trj.shape[0] != n_atoms:
                        trj = np.loadtxt(self.fnames[iind - shift], skiprows=9)

                        f = open(self.fnames[iind - shift],'r')
                        for i in range(3):
                            f.readline()
                        n_atoms = int(f.readline().split()[0])
                        f.close()

                        shift +=1 
                        
                    else:
                        flag = 1

            except:
                'empty dump files can be created due to storage limit'

                flag = 0
                shift = 1
                while (flag == 0):

                    try:
                        trj = np.loadtxt(self.fnames[iind - shift], skiprows=9)

                        f = open(self.fnames[iind - shift],'r')
                        for i in range(3):
                            f.readline()
                        n_atoms = int(f.readline().split()[0])
                        f.close()

                        if trj.shape[0] == n_atoms:
                            flag = 1
                        else:
                            shift += 1

                    except:
                        shift += 1

            density_tmp[iind, :] = self._computeLocalfC(trj)

            del trj

        if rank == 0:

            for iind in range(1, size):
                start_row = iind*avg_rows_per_process
                end_row = start_row + avg_rows_per_process
                if iind == size - 1:
                    end_row = len(self.fnames)

                recv = np.empty([len(self.fnames), self.n_bins - 1])
                req = self.comm.Irecv(recv, source=iind)
                req.Wait()

                density_tmp[start_row:end_row] = recv[start_row:end_row]
        else:
            send = density_tmp
            req = self.comm.Isend(send, dest=0)
            req.Wait()

        if rank == 0:
            density_initial = density_tmp[0]
            density_all = density_tmp[1:].reshape(len(self.source_dirs), len(self.freqs), self.Nrepeat, n_bins - 1)
            density_avg = np.empty([len(self.source_dirs), len(self.freqs), n_bins - 1])

            for dir_ind, source_dir in enumerate(self.source_dirs):
                for freq_ind, freq in enumerate(self.freqs):
                    density_avg[dir_ind, freq_ind, :] = density_all[dir_ind, freq_ind, :, :].mean(axis=0)

            density_avg = density_avg.reshape(-1, self.n_bins - 1)
            density_final = np.vstack((density_initial, density_avg))
            density_final = density_final.transpose()

            fig, ax = plt.subplots(figsize=(4, 3))
            #plt.rc('font',size=9)

            im = ax.imshow(density_final, vmin=0, vmax=1.0, cmap=cmaps['G2B'], origin='lower')
            ax.set_aspect('auto')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_ticks([0, 1.0])
            cbar.set_label(r'$f_{\mathrm{C}}$')
            xticks = np.linspace(0,len(self.source_dirs)*len(self.freqs),4)
            ax.set_xticks(xticks)
            ax.set_xticklabels([format(i*self.Nfreq*0.006*pow(10,-6), '.4f') for i in xticks])
            ax.set_xlabel(r'time ($\times 10^{6} \tau$)')
            ax.set_yticks(np.linspace(0, self.n_bins-1,5))
            ax.set_ylim(0 - 0.5, self.n_bins-1 + 0.5)
            ax.set_yticklabels(np.linspace(self.bins[0], self.bins[-1], 5))
            ax.set_ylabel(r'$z$ ($\sigma$)')
            plt.tight_layout(pad=1,h_pad=None,w_pad=None,rect=None)
            plt.savefig("fC_evol.png", dpi=1000)

            return density_final

    def _computeLocalfC(self, trj, **args):
        #trj_all = trj[trj[:,2] <= 4]
        trj_phase1 = trj[trj[:,2] == 1]
        trj_phase2 = trj[trj[:,2] == 3]

        hist1, bin_edges = np.histogram(trj_phase1[:,5], bins=self.bins)
        #density_1 = hist/(self.lx*self.ly*(bin_edges[1]-bin_edges[0]))
        hist2, bin_edges = np.histogram(trj_phase2[:,5], bins=self.bins)
        #density_2 = hist/(self.lx*self.ly*(bin_edges[1]-bin_edges[0]))

        #del trj_all
        del trj_phase1
        del trj_phase2

        fC = hist1/(hist1 + hist2)
        
        return fC

    def computeOrientation(self, zlo, zhi, n_bins, w, director):
        self.zlo = zlo
        self.zhi = zhi
        self.n_bins = n_bins
        self.w = w
        self.director = np.asarray(director)
        #self.bins = np.linspace(zlo, zhi, n_bins)

        size = self.comm.Get_size()
        rank = self.comm.Get_rank()

        if rank == 0:
            # number of bonds varies
            path = 'equil_0'
            files = [f for f in os.listdir(path) if 'data' in f]
            path_to_file = os.path.join(path, files[0])

            print('* Chain geometry info extracted')
            print('* path_to_file: {}'.format(path_to_file))

            f = open(path_to_file, 'r')

            ind = 0
            flag = 0

            while flag == 0:
                line = f.readline()
                match = re.search('^\d+ bonds', line)
                if match:
                    n_bonds = int(line.split()[0])

                match = re.search('^Atoms', line)
                if match:
                    flag = 1

                ind += 1

            f.close()
            print('* # of bonds = {}'.format(n_bonds))

            subprocess.Popen('grep -A {} "Bonds" {} > temp.txt'.format(n_bonds + 1, path_to_file), shell=True).wait()
            bonds = np.loadtxt('temp.txt', skiprows=2)
            os.remove('temp.txt')
            # choose bridge bonds between blocks
            #bonds = bonds[(bonds[:,1] == 3) | (bonds[:,1] == 6)].astype(int)
            bonds = bonds[bonds[:,1] == 3].astype(int)
            self.bonds = bonds

        else:
            bonds = None

        bonds = self.comm.bcast(bonds, root=0)
        self.bonds = bonds

        avg_rows_per_process = int(len(self.fnames)/size)

        start_row = rank * avg_rows_per_process
        end_row = start_row + avg_rows_per_process
        if rank == size - 1:
            end_row = len(self.fnames)

        #cos_tmp = np.empty([len(bonds), 2])
        S_tmp = np.empty([len(self.fnames), self.n_bins])
        for iind in range(start_row, end_row):

            try:

                trj = np.loadtxt(self.fnames[iind], skiprows=9)

                f = open(self.fnames[iind],'r')
                for i in range(3):
                    f.readline()
                n_atoms = int(f.readline().split()[0])
                f.close()

                flag = 0
                shift = 1
                while (flag == 0):
                    if trj.shape[0] != n_atoms:
                        trj = np.loadtxt(self.fnames[iind - shift], skiprows=9)

                        f = open(self.fnames[iind - shift],'r')
                        for i in range(3):
                            f.readline()
                        n_atoms = int(f.readline().split()[0])
                        f.close()

                        shift +=1 
                        
                    else:
                        flag = 1

            except:
                'empty dump files can be created due to storage limit'

                flag = 0
                shift = 1
                while (flag == 0):

                    try:
                        trj = np.loadtxt(self.fnames[iind - shift], skiprows=9)

                        f = open(self.fnames[iind - shift],'r')
                        for i in range(3):
                            f.readline()
                        n_atoms = int(f.readline().split()[0])
                        f.close()

                        if trj.shape[0] == n_atoms:
                            flag = 1
                        else:
                            shift += 1

                    except:
                        shift += 1

            trj = trj[np.argsort(trj[:,0])]

            cos_tmp = self.compute_cos(trj, iind)
            del trj

            tmp = self.compute_S(cos_tmp)
            S_tmp[iind, :] = tmp

        if rank == 0:
            S_1D = np.empty([len(self.fnames), self.n_bins])
            S_1D[start_row:end_row, :] = S_tmp[start_row:end_row]

            for iind in range(1, size):
                start_row = iind*avg_rows_per_process
                end_row = start_row + avg_rows_per_process
                if iind == size - 1:
                    end_row = len(self.fnames)

                recv = np.empty([len(self.fnames), self.n_bins])
                req = self.comm.Irecv(recv, source=iind)
                req.Wait()

                S_1D[start_row:end_row, :] = recv[start_row:end_row]
        else:
            send = S_tmp
            req = self.comm.Isend(send, dest=0)
            req.Wait()

        if rank == 0:

            op = Data.Data_TimeSeries2D(S_1D, self.Nrepeat)
            
            S_final = op.mean
            S_final = S_final.transpose()
            
            print(np.max(S_final), np.min(S_final))
            
            cmaps = {}
            from matplotlib.colors import ListedColormap
            #R = np.concatenate((np.linspace(0,1,85), np.ones(171)))
            #G = np.concatenate((np.linspace(0,1,85), np.linspace(1,0,171)))
            #B = np.concatenate((np.ones(85), np.linspace(1,0,171)))
            R = np.concatenate((np.linspace(0,1,5), np.ones(10)))
            G = np.concatenate((np.linspace(0,1,5), np.linspace(1,0,10)))
            B = np.concatenate((np.ones(5), np.linspace(1,0,10)))

            BWR = np.dstack((R, G, B))
            cmaps['BWR'] = ListedColormap(BWR[0], name='BWR')

            xticks = np.linspace(0,len(self.source_dirs)*len(self.freqs),4)

            fig, ax = plt.subplots(figsize=(4,3))
            im = ax.imshow(S_final, vmin=-0.5, vmax=1.0, cmap=cmaps['BWR'], origin='lower')
            ax.plot([xticks[0], xticks[-1]], [self.h, self.h], 'k--', lw=0.5)
            ax.set_aspect('auto')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_ticks([-0.5, 0, 1.0])
            cbar.set_label(r'$S$')
            ax.set_xlim(xticks[0] - 0.5, xticks[-1] + 0.5)
            ax.set_xticks(xticks)
            ax.set_xticklabels([format(i*self.Nfreq*0.006*pow(10,-6), '.4f') for i in xticks])
            ax.set_xlabel(r'time ($\times 10^{6} \tau$)')
            ax.set_yticks(np.linspace(0, self.n_bins-1,5))
            ax.set_ylim(0 - 0.5, self.n_bins-1 + 0.5)
            ax.set_yticklabels(np.linspace(self.zlo, self.zhi, 5))
            ax.set_ylabel(r'$z$ ($\sigma$)')
            plt.tight_layout(pad=1,h_pad=None,w_pad=None,rect=None)
            plt.savefig('angle_map.png', dpi=1000) 

            return S_final

    def computeChainAlignment(self, director, zlo=0, zhi=None):

        self.director = np.asarray(director)

        bins = np.linspace(0, 180, 72+1) # delta ranging from 0 to 180, delta_phi = 2.5
        #bins = np.linspace(0, 360, 144+1) # dleta ranging from 0 to 360, delta_phi = 2.5

        size = self.comm.Get_size()
        rank = self.comm.Get_rank()

        if rank == 0:
            if zhi == None:
                h = self.h
            else:
                h = zhi
        else:
            h = None
        
        h = self.comm.bcast(h, root=0)

        delta_h = 2 # thickness of bin
        ll_max = int(h - delta_h) # lower limit max

        if rank == 0:
            # number of bonds varies
            path = os.path.join(self.path, 'equil_0')
            files = [f for f in os.listdir(path) if 'data' in f]
            path_to_file = os.path.join(path, files[0])

            print('* Chain geometry info extracted')
            print('* path_to_file: {}'.format(path_to_file))

            f = open(path_to_file, 'r')

            ind = 0
            flag = 0

            while flag == 0:
                line = f.readline()
                match = re.search('^\d+ bonds', line)
                if match:
                    n_bonds = int(line.split()[0])

                match = re.search('^Atoms', line)
                if match:
                    flag = 1

                ind += 1

            f.close()
            print('* # of bonds = {}'.format(n_bonds))

            subprocess.Popen('grep -A {} "Bonds" {} > temp.txt'.format(n_bonds + 1, path_to_file), shell=True).wait()
            bonds = np.loadtxt('temp.txt', skiprows=2)
            os.remove('temp.txt')
            ## choose bridge bonds between blocks'
            bonds = bonds[(bonds[:,1] == 3) | (bonds[:,1] == 6)].astype(int)
            self.bonds = bonds

        else:
            bonds = None

        bonds = self.comm.bcast(bonds, root=0)
        self.bonds = bonds

        avg_rows_per_process = int(len(self.fnames)/size)

        start_row = rank * avg_rows_per_process
        end_row = start_row + avg_rows_per_process
        if rank == size - 1:
            end_row = len(self.fnames)

        angle_hist_tmp = np.empty((len(self.fnames), ll_max + 1, len(bins) - 1))
        for iind in range(start_row, end_row):

            trj = np.loadtxt(self.fnames[iind], skiprows=9)
            trj = trj[np.argsort(trj[:,0])]

            cos_tmp = self.compute_cos(trj, iind)

            for ind, ll in enumerate(np.linspace(0, ll_max, ll_max + 1, dtype=int)):
                ul = ll + delta_h

                cos_filtered = cos_tmp[np.logical_and(cos_tmp[:,0] > ll, cos_tmp[:,0] < ul)][:,1]
                angles = np.degrees(np.arccos(cos_filtered))

                hist, bin_edges = np.histogram(angles, bins)
                angle_hist_tmp[iind, ind, :] = hist
            bin_edges = bin_edges[:-1] + (bin_edges[1] - bin_edges[0])/2
        
        del trj

        if rank == 0:
            angle_hist = np.empty((len(self.fnames), ll_max + 1, len(bin_edges)))
            angle_hist[start_row:end_row, :, :] = angle_hist_tmp[start_row:end_row, :, :]

            for iind in range(1, size):
                start_row = iind*avg_rows_per_process
                end_row = start_row + avg_rows_per_process
                if iind == size - 1:
                    end_row = len(self.fnames)

                recv = np.empty((len(self.fnames), ll_max + 1, len(bin_edges)))
                req = self.comm.Irecv(recv, source=iind)
                req.Wait()

                angle_hist[start_row:end_row, :, :] = recv[start_row:end_row, :, :]
        else:
            send = angle_hist_tmp
            req = self.comm.Isend(send, dest=0)
            req.Wait()

        if rank == 0:

            result = {}
            result.update({'phi': bin_edges})

            align = Data.Data_TimeSeries3D(angle_hist, self.Nrepeat)

            print(align.mean[-1, 0, :])

            result.update({'hist_mean': align.mean})
            result.update({'hist_std': align.std})

            self.results.update({'chain_alignment': result})

    def compute_cos(self, trj, iind):
        '''
        Computes the position of the mid point of A-b-B chain
        and the angle made by the AB vector and the director
        '''
        
        ind_A = self.bonds[:,2] - 1
        ind_B = self.bonds[:,3] - 1

        #####
        a0 = trj[ind_A - 4]
        next_pt = trj[ind_A - 3]
        a1 = a0.copy()
        a1[:,3:] = self.d_pbc(next_pt, trj[ind_A - 4]) + trj[ind_A - 4,3:]
        next_pt = trj[ind_A - 2]
        a2 = a0.copy()
        a2[:,3:] = self.d_pbc(next_pt, trj[ind_A - 3]) + trj[ind_A - 3,3:]
        next_pt = trj[ind_A - 1]
        a3 = a0.copy()
        a3[:,3:] = self.d_pbc(next_pt, trj[ind_A - 2]) + trj[ind_A - 2,3:]
        next_pt = trj[ind_A - 0]
        a4 = a0.copy()
        a4[:,3:] = self.d_pbc(next_pt, trj[ind_A - 1]) + trj[ind_A - 1,3:]

        A = (a0 + a1 + a2 + a3 + a4)/5

        next_pt = trj[ind_A + 1]
        b0 = a0.copy()
        b0[:,3:] = self.d_pbc(next_pt, trj[ind_A - 0]) + trj[ind_A - 0,3:]
        next_pt = trj[ind_A + 2]
        b1 = a0.copy()
        b1[:,3:] = self.d_pbc(next_pt, trj[ind_A + 1]) + trj[ind_A + 1,3:]
        next_pt = trj[ind_A + 3]
        b2 = a0.copy()
        b2[:,3:] = self.d_pbc(next_pt, trj[ind_A + 2]) + trj[ind_A + 2,3:]
        next_pt = trj[ind_A + 4]
        b3 = a0.copy()
        b3[:,3:] = self.d_pbc(next_pt, trj[ind_A + 3]) + trj[ind_A + 3,3:]
        next_pt = trj[ind_A + 5]
        b4 = a0.copy()
        b4[:,3:] = self.d_pbc(next_pt, trj[ind_A + 4]) + trj[ind_A + 4,3:]

        B = (b0 + b1 + b2 + b3 + b4)/5
        #####

        #A = trj[ind_A - 3]
        #B = trj[ind_B + 3]
        
        #logic = np.logical_and(np.logical_and(A[:,4] > 5, B[:,4] > 5),(np.logical_and(A[:,4] < 25, B[:,4] < 25))) #
        #A = A[logic] #
        #B = B[logic] #

        d = self.d_pbc(A, B)
        z_avg = (trj[ind_A,5] + trj[ind_B,5])/2

        cos = np.empty((len(self.bonds), 4))
        #cos = np.empty((A.shape[0], 2)) #

        cos[:,0] = z_avg
        cos[:,1] = np.dot(d, self.director) / np.linalg.norm(d, axis=1)
        cos[:,2] = self.bonds[:,1]
        cos[:,3] = np.zeros(len(cos))
        cos[d[:,1] < 0, 3] = 1

        #np.savetxt('cos_{}.txt'.format(os.path.basename(self.fnames[iind]).split('dump.')[-1]), cos)

        return cos

    def compute_S(self, cos):
        '''
        Computes the Hermanns order parameter
        '''

        bin_starts = np.linspace(self.zlo, self.zhi - self.w, self.n_bins)
        bin_ends = bin_starts + self.w

        cos_binned = [cos[np.logical_and(cos[:,0] > bin_starts[i], cos[:,0] < bin_ends[i]), 1] for i in range(len(bin_starts))]

        S = np.empty(len(bin_starts))

        for bin_ind in range(len(cos_binned)):
            if len(cos_binned[bin_ind]) == 0:
                S[bin_ind] = np.nan
            else:
                S[bin_ind] = (3*np.mean(cos_binned[bin_ind]**2) - 1)/2

        return S

    def d_pbc(self, vector1, vector2):

        boxlength = [self.lx, self.ly]

        l_ref = [self.lx/2.0, self.ly/2.0]
        vector = vector1[:,3:] - vector2[:,3:]

        for i in [0, 1]:
            vector[vector[:,i] < (-1)*l_ref[i], i] += boxlength[i]
            vector[vector[:,i] > l_ref[i], i] -= boxlength[i]

        return vector

    def computeOrientationInPlane(self, zlo, zhi, n_bins, w, director):
        self.zlo = zlo
        self.zhi = zhi
        self.n_bins = n_bins
        self.w = w
        self.director = np.asarray(director)
        #self.bins = np.linspace(zlo, zhi, n_bins)

        size = self.comm.Get_size()
        rank = self.comm.Get_rank()

        if rank == 0:
            #number of bonds varies
            #subprocess.Popen('grep -A 182401 "Bonds" equil_0/data_preequil > temp.txt', shell=True).wait()
            subprocess.Popen('grep -A 276162 "Bonds" equil_0/data_preequil > temp.txt', shell=True).wait()

            bonds = np.loadtxt('temp.txt', skiprows=2)
            subprocess.Popen('rm temp.txt', shell=True).wait()
            bonds = bonds[(bonds[:,1] == 3) | (bonds[:,1] == 6)].astype(int)
            self.bonds = bonds

        else:
            bonds = None

        bonds = self.comm.bcast(bonds, root=0)
        self.bonds = bonds

        avg_rows_per_process = int(len(self.fnames)/size)

        start_row = rank * avg_rows_per_process
        end_row = start_row + avg_rows_per_process
        if rank == size - 1:
            end_row = len(self.fnames)

        S_tmp = np.empty([len(self.fnames), self.n_bins])
        for iind in range(start_row, end_row):

            try:

                trj = np.loadtxt(self.fnames[iind], skiprows=9)

                f = open(self.fnames[iind],'r')
                for i in range(3):
                    f.readline()
                n_atoms = int(f.readline().split()[0])
                f.close()

                flag = 0
                shift = 1
                while (flag == 0):
                    if trj.shape[0] != n_atoms:
                        trj = np.loadtxt(self.fnames[iind - shift], skiprows=9)

                        f = open(self.fnames[iind - shift],'r')
                        for i in range(3):
                            f.readline()
                        n_atoms = int(f.readline().split()[0])
                        f.close()

                        shift +=1 
                        
                    else:
                        flag = 1

            except:
                'empty dump files can be created due to storage limit'

                flag = 0
                shift = 1
                while (flag == 0):

                    try:
                        trj = np.loadtxt(self.fnames[iind - shift], skiprows=9)

                        f = open(self.fnames[iind - shift],'r')
                        for i in range(3):
                            f.readline()
                        n_atoms = int(f.readline().split()[0])
                        f.close()

                        if trj.shape[0] == n_atoms:
                            flag = 1
                        else:
                            shift += 1

                    except:
                        shift += 1

            trj = trj[np.argsort(trj[:,0])]

            cos_tmp = self.compute_cos_plane(trj)
            del trj

            tmp = self.compute_S_plane(cos_tmp)
            S_tmp[iind, :] = tmp

    def compute_cos_plane(self, trj):
        
        A = trj[self.bonds[:,2] - 1]
        B = trj[self.bonds[:,3] - 1]
        
        #logic = np.logical_and(np.logical_and(A[:,4] > 5, B[:,4] > 5),(np.logical_and(A[:,4] < 25, B[:,4] < 25))) #
        #A = A[logic] #
        #B = B[logic] #

        d = self.d_pbc(A, B)
        coords_avg = (A[:,3:6] + B[:,3:6])/2

        cos = np.empty([len(self.bonds), 4])
        #cos = np.empty([A.shape[0], 2]) #

        cos[:,:3] = coords_avg
        cos[:,3] = np.dot(d, self.director) / np.linalg.norm(d, axis=1)

        return cos

    def compute_S_plane(self, cos):

        bin_starts = np.linspace(self.zlo, self.zhi - self.w, self.n_bins)
        bin_ends = bin_starts + self.w

        binx = np.linspace(self.box[0][0], self.box[0][1], int((self.box[0][1]-self.box[0][0])/self.delta[0])+1)
        biny = np.linspace(self.box[1][0], self.box[1][1], int((self.box[1][1]-self.box[1][0])/self.delta[1])+1)

        S = np.zeros([len(bin_starts), binx-1, biny-1])

        for bin_ind in range(len(bin_starts)):
            cos_plane = cos[np.logical_and(cos[:,2] > bin_starts[bin_ind], cos[:,2] < bin_ends[bin_ind])] 

            cos_binned = stats.binned_statistic_dd(cos_plane[:,0:2], cos_plane[:,3]**2, statistic='mean', bins=[binx, biny]).statistic

            S[bin_ind,:,:] = (3*cos_binned - 1)/2
 
        return S

    def computeSegregation(self, blend, zlo, zhi, nx, ny, nz, w):
        self.blend = blend
        self.zlo = zlo
        self.zhi = zhi
        self.bins = [nx, ny, nz]
        self.w = w
        #self.bins = np.linspace(zlo, zhi, n_bins)

        size = self.comm.Get_size()
        rank = self.comm.Get_rank()

        avg_rows_per_process = int(len(self.fnames)/size)

        start_row = rank * avg_rows_per_process
        end_row = start_row + avg_rows_per_process
        if rank == size - 1:
            end_row = len(self.fnames)
        
        binary_3D =  {
                'A': np.empty([ny, nx, nz]),
                'B': np.empty([ny, nx, nz])
                }

        seg_tmp = np.empty([len(self.fnames), nz])

        for iind in range(start_row, end_row):
            tmp, delta = self.binarize_3D_n(self.fnames[iind])
            binary_3D['A'] = tmp['A']
            binary_3D['B'] = tmp['B']

            tmp = self._computesegregation(binary_3D)
            seg_tmp[iind, :] = tmp

        if rank == 0:
            seg_1D = np.empty([len(self.fnames), nz])
            seg_1D[start_row:end_row, :] = seg_tmp[start_row:end_row]

            for iind in range(1, size):
                start_row = iind*avg_rows_per_process
                end_row = start_row + avg_rows_per_process
                if iind == size - 1:
                    end_row = len(self.fnames)

                recv = np.empty([len(self.fnames), nz])
                req = self.comm.Irecv(recv, source=iind)
                req.Wait()

                seg_1D[start_row:end_row, :] = recv[start_row:end_row]
        else:
            send = seg_tmp
            req = self.comm.Isend(send, dest=0)
            req.Wait()

        if rank == 0:
            seg_initial = seg_1D[0]
            seg_all = seg_1D[1:].reshape(len(self.source_dirs), len(self.freqs), self.Nrepeat, nz)
            seg_avg = np.empty([len(self.source_dirs), len(self.freqs), nz])

            for dir_ind, source_dir in enumerate(self.source_dirs):
                for freq_ind, freq in enumerate(self.freqs):
                    seg_avg[dir_ind, freq_ind, :] = seg_all[dir_ind, freq_ind, :, :].mean(axis=0)

            seg_avg = seg_avg.reshape(-1, nz)
            seg_final = np.vstack((seg_initial, seg_avg))
            seg_final = seg_final.transpose()

            cmaps = {}
            from matplotlib.colors import ListedColormap
            G2B = np.dstack((np.linspace(128/255,1,256), np.linspace(0,1,256), np.linspace(128/255,0,256)))
            cmaps['G2B'] = ListedColormap(G2B[0], name='G2B')

            xticks = np.linspace(0,len(self.source_dirs)*len(self.freqs),4)

            fig, ax = plt.subplots(figsize=(4,3))
            im = ax.imshow(seg_final, vmin=0, vmax=0.5, cmap=cmaps['G2B'], origin='lower')
            ax.plot([xticks[0], xticks[-1]], [self.h, self.h], 'k--', lw=0.5)
            ax.set_aspect('auto')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_ticks([0, 0.5])
            cbar.set_label(r'phase separation')
            ax.set_xticks(xticks)
            ax.set_xticklabels([format(i*self.Nfreq*0.006*pow(10,-6), '.4f') for i in xticks])
            #yticks = np.arange(nz)
            ax.set_yticks(np.linspace(0, self.bins[2]-1,5))
            ax.set_ylim(0 - 0.5, self.bins[2]-1 + 0.5)
            ax.set_yticklabels(np.linspace(self.zlo, self.zhi, 5))
            #ax.set_yticklabels(["{:.1f}".format((i+1)*delta[2]) for i in yticks])
            ax.set_xlabel(r'time ($\times 10^{6} \tau$)')
            ax.set_ylabel(r'$z$ ($\sigma$)')
            plt.tight_layout(pad=1,h_pad=None,w_pad=None,rect=None)
            plt.savefig('local_z_seg.png', dpi=1000)

            return seg_final

    def _computeSegregation(self, binary_3D):

        seg = binary_3D['A'] / (binary_3D['A'] + binary_3D['B'])
        seg[binary_3D['A'] + binary_3D['B'] == 0] = np.nan
        seg[(binary_3D['A'] == 0) & (binary_3D['B'] != 0)] = 0

        seg_1D = seg.reshape(-1)

        seg_1D = seg_1D[~np.isnan(seg_1D)]

        fA_bins = np.linspace(0, 1, 21)
        hist, bin_edges = np.histogram(seg_1D, bins=fA_bins)
        fA = bin_edges[1:] - (bin_edges[1] - bin_edges[0])/2

        fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(fA, hist)
        plt.savefig('fA.png', dpi=500)  

        fA_evol = hist/len(seg_1D)

        seg_2D = seg.reshape(-1, seg.shape[2])

        fA_z = np.zeros(seg_2D.shape[1])
        for iind in range(seg_2D.shape[1]):
            tmp = seg_2D[:, iind]
            res = tmp[~np.isnan(tmp)]
            fA_z[iind] = np.std(res)

        return fA_z

    def binarize_3D_n(self, fname):
        # Discretize simulation box into voxels and identifies a voxel at which each bead located
        trj = np.loadtxt(fname, skiprows=9)
        trj = trj[np.argsort(trj[:,0])]
        trj[:,5] = trj[:,5] - self.box[2][0]

        binary = {}
        binary.update(
                {'A': np.zeros([
                    self.bins[1],
                    self.bins[0],
                    self.bins[2]
                    ])}
                )

        binary.update(
                {'B': np.zeros([
                    self.bins[1],
                    self.bins[0],
                    self.bins[2]
                    ])}
                )

        delta = [0, 0, 0]
        delta[0] = self.lx/self.bins[0]
        delta[1] = self.ly/self.bins[1] 
        delta[2] = self.w

        #print(delta)

        trjs_filter = {}
        if self.blend:
            trjs_filter.update(
                    {'A': trj[(trj[:,2] == 1) | (trj[:,2] == 3)]}
                    )
            trjs_filter.update(
                    {'B': trj[(trj[:,2] == 2) | (trj[:,2] == 4)]}
                    )
        else:
            trjs_filter.update(
                    {'A': trj[trj[:,2] == 1]}
                    )
            trjs_filter.update(
                    {'B': trj[trj[:,2] == 2]}
                    ) 

        bin_starts = np.linspace(self.zlo, self.zhi - self.w, self.bins[2])
        bin_ends = bin_starts + self.w

        for key in trjs_filter:
            trj_filter = trjs_filter[key]

            for iind in range(len(bin_starts)):
                tmp = trj_filter[np.logical_and(trj_filter[:, 5] > bin_starts[iind], trj_filter[:, 5] < bin_ends[iind])]

                q = np.zeros(3, dtype=int)

                q[2] = iind
                for jind in range(len(tmp)):
                    q[0] = int((tmp[jind,4]%(self.box[1][1]-self.box[1][0]))//delta[1])
                    q[1] = int((tmp[jind,3]%(self.box[0][1]-self.box[0][0]))//delta[0])
                
                    binary[key][q[0], q[1], q[2]] += 1

        return binary, delta

    def computeChisq(self, blend, delta, run_args, **kwargs):

        self.blend = blend
        self.delta = delta

        if 'remain' in run_args:
            if run_args['remain'] == 'A':
                self.phase = [i[0] for i in run_args['phase']]
            else:
                self.phase = [i[1] for i in run_args['phase']]

        size = self.comm.Get_size()
        rank = self.comm.Get_rank()

        if rank == 0:
 
            f = open(self.fnames[0],'r')
            box = []
            for i in range(5):
                f.readline()
            for i in range(3):
                line = f.readline().split()
                box.append([float(line[0]), float(line[1])])
            f.close()
            self.box = box

            trj = np.loadtxt(self.fnames[0], skiprows=9)
            trj = trj[np.argsort(trj[:,0])]
            trj[:,5] = trj[:,5] - self.box[2][0]

            topview_initial = self.view_xy(self.binarize_3D(trj))

            del trj

            dims = {'h': topview_initial.shape[0], 'w': topview_initial.shape[1]}
            self.dims = dims

        else:
            box = None
            dims = None

        box = self.comm.bcast(box, root=0)
        dims = self.comm.bcast(dims, root=0)
        self.box = box
        self.dims = dims
 
        avg_rows_per_process = int(len(self.fnames)/size)
        
        start_row = rank * avg_rows_per_process
        end_row = start_row + avg_rows_per_process
        if rank == size - 1:
            end_row = len(self.fnames)

        #topview = np.empty([len(fnames), self.dims['h'], self.dims['w']])
        topview = np.empty([len(self.fnames), self.dims['h'] * self.dims['w']])

        for iind in range(start_row, end_row):

            try:

                trj = np.loadtxt(self.fnames[iind], skiprows=9)

                f = open(self.fnames[iind],'r')
                for i in range(3):
                    f.readline()
                n_atoms = int(f.readline().split()[0])
                f.close()

                #print(n_atoms)

                flag = 0
                shift = 1
                while (flag == 0):
                    if trj.shape[0] != n_atoms:
                        trj = np.loadtxt(self.fnames[iind - shift], skiprows=9)

                        f = open(self.fnames[iind - shift],'r')
                        for i in range(3):
                            f.readline()
                        n_atoms = int(f.readline().split()[0])
                        f.close()

                        shift +=1 
                        
                    else:
                        flag = 1

            except:
                'empty dump files can be created due to storage limit'

                flag = 0
                shift = 1
                while (flag == 0):

                    try:
                        trj = np.loadtxt(self.fnames[iind - shift], skiprows=9)

                        f = open(self.fnames[iind - shift],'r')
                        for i in range(3):
                            f.readline()
                        n_atoms = int(f.readline().split()[0])
                        f.close()

                        if trj.shape[0] == n_atoms:
                            flag = 1
                        else:
                            shift += 1

                    except:
                        shift += 1

            trj = trj[np.argsort(trj[:,0])]
            trj[:,5] = trj[:,5] - self.box[2][0]

            topview[iind] = self.view_xy(self.binarize_3D(trj)).reshape(-1) #with custom bin

            del trj

        if rank == 0:

            for iind in range(1, size):
                start_row = iind*avg_rows_per_process
                end_row = start_row + avg_rows_per_process
                if iind == size - 1:
                    end_row = len(self.fnames)

                #recv = np.empty([len(fnames), self.dims['h'], self.dims['w']])
                recv = np.empty([len(self.fnames), self.dims['h'] * self.dims['w']])
                req = self.comm.Irecv(recv, source=iind)
                req.Wait()

                topview[start_row:end_row] = recv[start_row:end_row]

                del recv

        else:
            #send = np.empty([len(fnames), self.dims['h'], self.dims['w']])
            send = np.empty([len(self.fnames), self.dims['h'] * self.dims['w']])
            for iind in range(start_row, end_row):
                send[iind] = topview[iind]
            req = self.comm.Isend(send, dest=0)
            req.Wait()

            del send

        if rank == 0:
            topview = topview[1:].reshape(len(self.fnames)-1, self.dims['h'], self.dims['w'])
            topview_all = topview.copy().reshape(len(self.source_dirs), len(self.freqs), self.Nrepeat, self.dims['h']*self.dims['w'])
            del topview
            topview_avg = np.empty([len(self.source_dirs), len(self.freqs), self.dims['h']*self.dims['w']])

            for dir_ind, source_dir in enumerate(self.source_dirs):
                for freq_ind, freq in enumerate(self.freqs):
                    topview_avg[dir_ind, freq_ind, :] = topview_all[dir_ind, freq_ind, :, :].mean(axis=0)

            topview_avg = topview_avg.reshape(-1, self.dims['h']*self.dims['w'])
            topview_final = np.empty([topview_avg.shape[0]+1, topview_avg.shape[1]])
            topview_final[0] = topview_initial.copy().reshape(-1)
            topview_final[1:] = topview_avg.copy()

            del topview_initial
            del topview_avg

        else:
            topview_final = None

        topview_final = self.comm.bcast(topview_final, root=0)

        avg_rows_per_process = int(len(topview_final)/size)
        
        start_row = rank * avg_rows_per_process
        end_row = start_row + avg_rows_per_process
        if rank == size - 1:
            end_row = len(topview_final)

        chisq = np.empty(len(topview_final))

        for iind in range(start_row, end_row):
            chisq[iind] = self._computeChisq(topview_final[iind].reshape(self.dims['h'], self.dims['w']), iind, **run_args)

        if rank == 0:

            for iind in range(1, size):
                start_row = iind*avg_rows_per_process
                end_row = start_row + avg_rows_per_process
                if iind == size - 1:
                    end_row = len(topview_final)

                recv = np.empty(len(topview_final))
                req = self.comm.Irecv(recv, source=iind)
                req.Wait()

                chisq[start_row:end_row] = recv[start_row:end_row]
        else:
            send = chisq
            req = self.comm.Isend(send, dest=0)
            req.Wait()

        if rank == 0:
            for iind, val in enumerate(chisq):
                print(iind, val)

            np.savetxt('chisq.txt', chisq)

            '''
            fig, ax = plt.subplots(figsize=(4,3))
            ax.plot(chisq)
            #ax.set_aspect('auto')
            divider = make_axes_locatable(ax)
            xticks = np.linspace(0,len(self.source_dirs)*len(self.freqs),4)
            ax.set_xticks(xticks)
            ax.set_xticklabels([format(i*self.Nfreq*0.006*pow(10,-6), '.4f') for i in xticks])
            ax.set_xlabel(r'time ($\times 10^{6} \tau$)')
            ax.set_ylabel(r'$\chi^{2}$')
            ax.set_ylim(0, 1000000)
            plt.tight_layout(pad=1,h_pad=None,w_pad=None,rect=None)
            plt.savefig('chisq.png', dpi=1000)
            '''

    def binarize_3D(self, trj):

        '''
        binary = np.empty([
            int((self.box[1][1] - self.box[1][0])/self.delta[1]),
            int((self.box[0][1] - self.box[0][0])/self.delta[0]),
            int(math.ceil(max(trj[:,5])/self.delta[2]))
            ])

        if self.blend:
            trj_filter = trj[(trj[:,2] == self.phase[0]) | (trj[:,2] == self.phase[1])]
        else:
            trj_filter = trj[trj[:,2] == self.phase[0]]

        for iind in range(len(trj_filter)):
            q = np.zeros(3, dtype=int)
            q[0] = int((trj_filter[iind,4]%(self.box[1][1]-self.box[1][0]))//self.delta[1])
            q[1] = int((trj_filter[iind,3]%(self.box[0][1]-self.box[0][0]))//self.delta[0])
            q[2] = int((trj_filter[iind,5]%(self.box[2][1]-self.box[2][0]))//self.delta[2])
            binary[q[0], q[1], q[2]] = 1
        '''
        delta = self.delta
        box = self.box
        blend = self.blend

        binx = np.linspace(box[0][0], box[0][1], int((box[0][1]-box[0][0])/delta[0])+1)
        biny = np.linspace(box[1][0], box[1][1], int((box[1][1]-box[1][0])/delta[1])+1)
        binz = np.linspace(0, math.ceil(max(trj[:,5])), int(math.ceil(max(trj[:,5]))/delta[2])+1)

        if blend:
            # bead types
            trj_filter = trj[(trj[:,2] == 1) | (trj[:,2] == 3)]
        else:
            trj_filter = trj[trj[:,2] == 1]

        binary = stats.binned_statistic_dd(trj_filter[:,[4, 3, 5]], np.ones(trj_filter.shape[0]), statistic='count', bins=[biny, binx, binz]).statistic

        return binary

    def view_xy(self, binary_3D):

        array = binary_3D

        'sum up over thickness'
        topview = np.sum(binary_3D, axis=2)

        'averagve over thickness'
        #topview = np.average(binary_3D, axis=2)

        return topview

    def _computeChisq(self, topview, num, **kwargs):

        fig=plt.figure(figsize=(4,3))
        ax=fig.add_subplot(111)
        ax.imshow(topview, cmap=plt.get_cmap('gray'), origin='lower')
        plt.tight_layout(pad=1, h_pad=None, w_pad=None, rect=None)
        plt.savefig('{}_topview.png'.format(num), dpi=1000)
        plt.close()

        self.scale = kwargs['distance_in_pixels'] / kwargs['known_distance']

        fft_resolution = kwargs['fft_resolution']
        q_del_px = 2*math.pi/fft_resolution
        q_del = q_del_px*self.scale

        q = np.linspace(-1.5, 1.5, 7)
        qq_x = fft_resolution//2 + q/q_del
        qq_y = fft_resolution//2 + q/q_del

        fft = np.fft.fft2(topview)
        fshift = np.fft.fftshift(fft)
        magnitude_spectrum_shift = np.abs(fshift)
        #magnitude_spectrum_shift = 20*np.log(np.abs(fshift))
        #magnitude_spectrum_shift = np.log(np.abs(fshift))

        x = np.linspace(0, self.dims['w']-2, self.dims['w']-1)
        y = np.linspace(0, self.dims['h']-2, self.dims['h']-1)
        z = magnitude_spectrum_shift[1:self.dims['h'],1:self.dims['w']]

        from scipy import interpolate
        f = interpolate.interp2d(x, y, z, kind='linear')

        xnew = np.linspace(0,self.dims['w']-2,fft_resolution+1)
        ynew = np.linspace(0,self.dims['h']-2,fft_resolution+1)
        znew = f(xnew, ynew)

        fig=plt.figure(figsize=(4,3))
        ax=fig.add_subplot(111)
        ax.imshow(znew, cmap=plt.get_cmap('jet'), origin='lower')
        ax.set_xlim(qq_x[0], qq_x[-1])
        ax.set_ylim(qq_y[0], qq_y[-1])
        ax.set_xticks(qq_x)
        ax.set_xticklabels(q)
        ax.set_yticks(qq_y)
        ax.set_yticklabels(q)
        ax.set_xlabel(r'$q_{x}$ (1/$\sigma$)')
        ax.set_ylabel(r'$q_{y}$ (1/$\sigma$)')
        plt.tight_layout(pad=1, h_pad=None, w_pad=None, rect=None)
        plt.savefig('{}_2Dfft.png'.format(num), dpi=500)
        #plt.savefig(os.path.join(self.path, 'results', 'MD_fft_interpolated.png'), dpi=1000)
        plt.close()

        " Calculate average in the radial direction "
        zoom_mag = kwargs['zoom_mag']
        c_x = znew.shape[1]//(2*zoom_mag)
        c_y = znew.shape[0]//(2*zoom_mag)

        x, y = np.meshgrid(np.arange(fft_resolution//(1*zoom_mag)), np.arange(fft_resolution//(1*zoom_mag)))
        R = np.sqrt((x-c_x)**2 + (y-c_y)**2)

        ll = fft_resolution//2 - fft_resolution//(2*zoom_mag)
        ul = fft_resolution//2 + fft_resolution//(2*zoom_mag)
        znew_zoomed = znew[ll:ul, ll:ul]

        f = lambda r : znew_zoomed[(R >= r-.5) & (R < r+.5)].mean()
        r = np.linspace(1, fft_resolution//(2*zoom_mag), fft_resolution//(2*zoom_mag))
        mean = np.vectorize(f)(r)
        mean *= r**3

        r *= self.scale # Scale r to convert distance to pixel

        " Fit a gaussian plus line curve to I-q "
        left_ind = 30#30
        right_ind = 200#120
        x = r[left_ind:right_ind]
        y = mean[left_ind:right_ind]

        import lmfit

        pars = lmfit.Parameters()
        pars.add_many(('amp', max(y)), ('cen', np.average(x)), ('wid', np.std(x)), ('slope', 0), ('intercept', 0))

        def gaussian_plus_line(v, x):
            """ line + 1-d gaussian """

            gauss = v['amp'] * np.exp(-(x-v['cen'])**2 / v['wid'])
            line = v['slope']*x + v['intercept']
            return gauss + line

        def func2minimize(pars, x, y):

            v = pars.valuesdict()
            m = gaussian_plus_line(v, x)

            return m - y

        try:
            mi = lmfit.minimize(func2minimize, pars, args=(x, y))
            lmfit.printfuncs.report_fit(mi.params, min_correl=0.5)

            #print(mi.chisqr)

            l0 = 2 * math.pi / (mi.params['cen'] * q_del_px)# / scale #if r not scaled, divide by scale
            l0_minus_sigma = 2 * math.pi / ((mi.params['cen'] + mi.params['cen'].stderr) * q_del_px)
            #print ('L0 = %.4f' % l0)
            #print ('L0_std = %.4f' % (l0 - l0_minus_sigma))

            fig, ax = plt.subplots(figsize=(4,3))
            ax.plot(r*q_del_px, mean, 'o', mec='None', ms=2)
            ax.plot(x*q_del_px, gaussian_plus_line(mi.params, x), 'r-')
            ax.set_xlabel(r'$q$ (1/px)')
            ax.set_ylabel(r'$I$ (a.u.)')
            plt.tight_layout(pad=1, h_pad=None, w_pad=None, rect=None)
            plt.savefig('{}_1Dfft_fit.png'.format(num), dpi=500)
            #plt.savefig(os.path.join(self.path, 'results', 'Iq.png'), dpi=1000)
            plt.close()

            #self.l0 = l0
            #self.l0_std = l0 - l0_minus_sigma

            return mi.chisqr

        except:
            print("Could not fit a curve")
            self.l0 = 15.0
            self.l0_std = 0.0

    def position(self):

        #moi
        #molnumber, N, fA
        mol_of_interest = np.loadtxt('moi.txt')

        size = self.comm.Get_size()
        rank = self.comm.Get_rank()

        avg_rows_per_process = int(len(self.fnames)/size)

        start_row = rank * avg_rows_per_process
        end_row = start_row + avg_rows_per_process
        if rank == size - 1:
            end_row = len(self.fnames)

        position_z = np.empty([len(self.fnames), mol_of_interest.shape[0]])

        if rank == 0:

            for iind in range(1, size):
                start_row = iind*avg_rows_per_process
                end_row = start_row + avg_rows_per_process
                if iind == size - 1:
                    end_row = len(self.fnames)

                recv = np.empty([len(self.fnames), mol_of_interest.shape[0]])
                req = self.comm.Irecv(recv, source=iind)
                req.Wait()

                position_z[start_row:end_row] = recv[start_row:end_row]
        else:
            send = position_z
            req = self.comm.Isend(send, dest=0)
            req.Wait()

        if rank == 0:
            position_z_initial = position_z[0]
            position_z_all = position_z[1:].reshape(len(self.source_dirs), len(self.freqs), self.Nrepeat, mol_of_interest.shape[0])
            position_z_avg = np.empty([len(self.source_dirs), len(self.freqs), mol_of_interest.shape[0]])

            for dir_ind, source_dir in enumerate(self.source_dirs):
                for freq_ind, freq in enumerate(self.freqs):
                    position_z_avg[dir_ind, freq_ind, :] = position_z_all[dir_ind, freq_ind, :, :].mean(axis=0)

            position_z_avg = position_z_avg.reshape(-1, mol_of_interest.shape[0])
            position_z_final = np.vstack((position_z_initial, position_z_avg))

            fig, ax = plt.subplots(figsize=(4, 3))
            #plt.rc('font',size=9)

            ax.plot(position_z_final[:, 0], position_z_final[:, 1])
            ax.set_aspect('auto')
            xticks = np.linspace(0,len(self.source_dirs)*len(self.freqs),4)
            ax.set_xticks(xticks)
            ax.set_xticklabels([format(i*self.Nfreq*0.006*pow(10,-6), '.4f') for i in xticks])
            ax.set_xlabel(r'time ($\times 10^{6} \tau$)')
            #ax.set_yticks(np.linspace(0, self.n_bins-1,5))
            #ax.set_ylim(0 - 0.5, self.n_bins-1 + 0.5)
            #ax.set_yticklabels(np.linspace(self.bins[0], self.bins[-1], 5))
            #ax.set_ylabel(r'$z$ ($\sigma$)')
            plt.tight_layout(pad=1,h_pad=None,w_pad=None,rect=None)
            plt.savefig("position.png", dpi=1000)

    def concentration(self, lx, ly, z_max, phase):

        size = self.comm.Get_size()
        rank = self.comm.Get_rank()

        avg_rows_per_process = int(len(self.fnames)/size)

        start_row = rank * avg_rows_per_process
        end_row = start_row + avg_rows_per_process
        if rank == size - 1:
            end_row = len(self.fnames)

        binz = np.linspace(0, z_max, z_max+1)
        concentration_tmp = np.empty((len(self.fnames), z_max))
        for iind in range(start_row, end_row):

            try:

                trj = np.loadtxt(self.fnames[iind], skiprows=9)

                f = open(self.fnames[iind],'r')
                for i in range(3):
                    f.readline()
                n_atoms = int(f.readline().split()[0])
                f.close()

                #print(n_atoms)

                flag = 0
                shift = 1
                while (flag == 0):
                    if trj.shape[0] != n_atoms:
                        trj = np.loadtxt(self.fnames[iind - shift], skiprows=9)

                        f = open(self.fnames[iind - shift],'r')
                        for i in range(3):
                            f.readline()
                        n_atoms = int(f.readline().split()[0])
                        f.close()

                        shift +=1 
                        
                    else:
                        flag = 1

            except:
                'empty dump files can be created due to storage limit'

                flag = 0
                shift = 1
                while (flag == 0):

                    try:
                        trj = np.loadtxt(self.fnames[iind - shift], skiprows=9)

                        f = open(self.fnames[iind - shift],'r')
                        for i in range(3):
                            f.readline()
                        n_atoms = int(f.readline().split()[0])
                        f.close()

                        if trj.shape[0] == n_atoms:
                            flag = 1
                        else:
                            shift += 1

                    except:
                        shift += 1

            if phase == 'C':
                logic = trj[:,2] < 3
            elif phase == 'L':
                logic = np.logical_and(trj[:,2] > 2, trj[:,2] < 5)
            concentration_tmp[iind] = stats.binned_statistic_dd(trj[logic,5], None, statistic='count', bins=[binz]).statistic

            del trj

        if rank == 0:

            for iind in range(1, size):
                start_row = iind*avg_rows_per_process
                end_row = start_row + avg_rows_per_process
                if iind == size - 1:
                    end_row = len(self.fnames)

                recv = np.empty((len(self.fnames), z_max))
                req = self.comm.Irecv(recv, source=iind)
                req.Wait()

                concentration_tmp[start_row:end_row] = recv[start_row:end_row]
        else:
            send = concentration_tmp
            req = self.comm.Isend(send, dest=0)
            req.Wait()

        if rank == 0:
            concentration_initial = concentration_tmp[0]
            concentration_all = concentration_tmp[1:].reshape(len(self.source_dirs), len(self.freqs), self.Nrepeat, z_max)
            concentration_avg = np.empty((len(self.source_dirs), len(self.freqs), z_max))

            for dir_ind, source_dir in enumerate(self.source_dirs):
                for freq_ind, freq in enumerate(self.freqs):
                    concentration_avg[dir_ind, freq_ind, :] = concentration_all[dir_ind, freq_ind, :, :].mean(axis=0)

            concentration_avg = concentration_avg.reshape(-1, z_max)
            concentration_final = np.vstack((concentration_initial, concentration_avg))
            concentration_final = concentration_final.transpose()
            concentration_final /= lx*ly

            fig, ax = plt.subplots(figsize=(4,3))
            ax.set_aspect('equal', adjustable='box')
            ax.imshow(concentration_final, origin='lower')
            plt.tight_layout(pad=1,h_pad=None,w_pad=None,rect=None)
            plt.savefig('result.png', dpi=500)
            plt.close()

            t = np.linspace(0, concentration_final.shape[1]-1, concentration_final.shape[1])
            tt, zz = np.meshgrid(t-0.5, np.linspace(0, z_max-1, z_max))
            concentration_ = concentration_final.ravel()
            t_ = tt.ravel()
            z_ = zz.ravel()

            grad = np.gradient(concentration_final, t, axis=1)
            grad_ = grad.ravel()

            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.bar3d(t_, z_, 0, 1, 1, concentration_)
            #ax.plot_surface(tt, zz, concentration_final, cmap='viridis', edgecolor='None')
            #ax.contour3D(tt, zz, concentration_final, 50)
            ax.view_init(30, 120)
            xticks = np.linspace(0, len(self.source_dirs)*len(self.freqs), 4)
            xticklabels = [format(i*self.Nfreq*0.006*pow(10,-6), '.4f') for i in xticks]
            ax.set_xlim(xticks[-1]+0.5, xticks[0]-0.5)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel(r'$t/\mathrm{M}\tau$')
            ax.set_ylim(0-0.5, 69+0.5)
            #ax.set_ylim(69+.5, 0-0.5)
            ax.set_yticks(np.linspace(0, 70, 5)) 
            ax.set_ylabel(r'$z/\sigma$')
            ax.set_zlim(0, 0.85)
            ax.set_zlabel(r'concentration/$\sigma^{-3}$')
            plt.savefig('result_3d.png', dpi=500)

            '''  
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            from matplotlib.colors import ListedColormap
            import matplotlib.colors as colors
            R = np.concatenate((np.linspace(0,1,128), np.ones(128)))
            G = np.concatenate((np.linspace(0,1,128), np.linspace(1,0,128)))
            B = np.concatenate((np.ones(128), np.linspace(1,0,128)))
            BWR = np.dstack((R, G, B))
            cmaps = {}
            cmaps['BWR'] = ListedColormap(BWR[0], name='BWR')

            fig = plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            #ax.bar3d(t_, z_, 0, 1, 1, grad_)
            #ax.view_init(30, 120)
            #im = ax.imshow(grad, origin='lower', vmin=-0.5, vmax=0.5, cmap=cmaps['BWR'])
            im = ax.imshow(grad, norm=colors.SymLogNorm(linthresh=0.001, linscale=0.001, vmin=-0.5, vmax=0.5, base=10), origin='lower', cmap=cmaps['BWR'])
            ax.set_aspect('auto')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_ticks([-0.1, -0.01, -0.001, 0.01, 0.1])
            cbar.set_ticklabels([r'$-10^{-1}$', r'$-10^{-2}$', r'$-10^{\pm3}$', r'$10^{-2}$', r'$10^{-1}$'])
            cbar.set_label(r'slope')
            xticks = np.linspace(0, len(self.source_dirs)*len(self.freqs), 4)
            xticklabels = [format(i*self.Nfreq*0.006*pow(10,-6), '.4f') for i in xticks]
            ax.set_xlim(xticks[0]-0.5, xticks[-1]+0.5)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel(r'$t/\mathrm{M}\tau$')
            ax.set_ylim(0-0.5, 69+0.5)
            ax.set_yticks(np.linspace(0, 70, 5)) 
            ax.set_ylabel(r'$z/\sigma$')
            ax.set_ylabel(r'$z/\sigma$')
            #ax.set_zlabel(r'concentration/$\sigma^{-3}$')
            plt.tight_layout(pad=1,h_pad=None,w_pad=None,rect=None)
            plt.savefig('result_grad_3d.png', dpi=500)
            '''

            save = {}
            save.update({'tt': tt})
            save.update({'zz': zz})
            save.update({'concentration': concentration_final})
            save.update({'xticks': xticks})
            save.update({'xticklabels': xticklabels})
            
            np.save('save.npy', save, allow_pickle=True)

    def computeObjsize(self, blend, delta):
        '''
        Carry out film resconstruction by looking at every height.
        With the matrix flood-filled, calculate the size of each morphological object.
        '''
        from scipy.ndimage import gaussian_filter

        sys.setrecursionlimit(100000)

        self.blend = blend
        self.delta = delta
        size = self.comm.Get_size()
        rank = self.comm.Get_rank()

        fnames = self.fnames.copy()
        fnames.pop(0)

        fnames = [fnames[self.Nrepeat*ind: self.Nrepeat*(ind + 1)] for ind in range(int(len(fnames)/self.Nrepeat))]
        
        avg_rows_per_process = int(len(fnames)/size)

        start_row = rank * avg_rows_per_process
        end_row = start_row + avg_rows_per_process
        if rank == size - 1:
            end_row = len(fnames)

        if rank == 0:
            h = self.h
        else:
            h = None

        h = self.comm.bcast(h, root=0)

        delta_h = 2
        ll_max = int(h - delta_h)

        res = np.empty((len(fnames), ll_max + 1, 3))
        for row_ind in range(start_row, end_row):

            for f_ind, fname in enumerate(fnames[row_ind]):
                trj = np.loadtxt(fname, skiprows=9)

                if f_ind == 0:
                    binary_3D = self.binarize_3D(trj)
                    binary_3D[binary_3D > 0] = 1
                else:
                    new_binary_3D = self.binarize_3D(trj)
                    new_binary_3D[new_binary_3D > 0] = 1
                    dim_z = min(binary_3D.shape[2], new_binary_3D.shape[2])
                    binary_3D = binary_3D[:, :, :dim_z] + new_binary_3D[:, :, :dim_z]

            del trj

            for ind, ll in enumerate(np.linspace(0, ll_max, ll_max + 1, dtype=int)):
                ul = ll + delta_h

                binary_2D = binary_3D[:, :, ll*2: ul*2].sum(axis=2)

                binary_2D[binary_2D > 0] = 1
                binary_2D = gaussian_filter(binary_2D, 1)
                binary_2D[binary_2D > 0.5] = 1
                binary_2D[binary_2D < 0.5] = 0  

                'film reconstruction snapshots'
                plt.figure()
                plt.imshow(binary_2D, origin='lower')
                plt.savefig('test_{:02d}.png'.format(ind))
                plt.close()

                object_size = []
                try:
                    blobs = group2blob(binary_2D)

                    h, w = binary_2D.shape
    
                    for blob_ind, blob in enumerate(blobs):
                        #print (len(blob))
                        if len(blob) < 10:
                            for j in range(len(blob)):
                                binary_2D[blob[j][0]%h, blob[j][1]%w]=0
                        else:
                            object_size.append(len(blob))
                except:
                    object_size.append(0)
                
                #print(np.mean(object_size), np.std(object_size))

                res[row_ind, ind, :] = [(ll + ul)/2, np.mean(object_size), np.std(object_size)]

        if rank == 0:

            for iind in range(1, size):
                start_row = iind*avg_rows_per_process
                end_row = start_row + avg_rows_per_process
                if iind == size - 1:
                    end_row = len(self.fnames)

                recv = np.empty((len(fnames), ll_max + 1 , 3))
                req = self.comm.Irecv(recv, source=iind)
                req.Wait()

                res[start_row:end_row] = recv[start_row:end_row]
        else:
            send = res
            req = self.comm.Isend(send, dest=0)
            req.Wait()

        if rank == 0:

            for ind, val in enumerate(res): 
                plt.figure()
                plt.errorbar(val[:,0], val[:,1], yerr=val[:,2])
                plt.ylim(0, 2500)
                plt.savefig('test2_{:02d}.png'.format(ind))
                plt.close()

    def computeStructurefactor(self, types, n, L=None):

        if L is None:
            L = max(self.lx, self.ly)

        trj = np.loadtxt(self.fnames[0], skiprows=9)
        trj = trj[np.argsort(trj[:,0])]
        if len(types)==1:
            logic = trj[:,2]==types[0]
        else:
            logic = trj[:,2]==types[0]
            for i in range(1,len(types)):
                logic = np.logical_or(logic, trj[:,2]==types[i])
        num_atoms = len(trj[logic])

        f = open('inc.parameters', 'w')
        f.write('C parameters\n')
        f.write('        parameter (num_atoms=%d)\n' %num_atoms)
        f.write('        parameter (n=%d)\n' %n)
        f.write('        parameter (L=%f)\n' %L)
        f.close()

        for ind, fname in enumerate(self.fnames):
            trj = np.loadtxt(fname, skiprows=9)
            trj = trj[np.argsort(trj[:,0])]
            trj = trj[logic]

            f = open('dump_temp', 'w')
            f.write('ITEM: TIMESTEP\n')
            f.write('0\n')
            f.write('ITEM: NUMBER OF ATOMS\n')
            f.write('0\n')
            f.write('ITEM: BoX BOUNDS pp pp ff\n')
            f.write('0 0\n')
            f.write('0 0\n')
            f.write('0 0\n')
            f.write('ITEM: ATOMS id mol type x y z\n')
            for line in trj:
                f.write('%d\t%d\t%d\t%s\t%s\t%s\n' %(line[0], line[1], line[2], line[3], line[4], line[5]))
            f.close()
            
            subprocess.Popen('mpif90 sf_mpi.f', shell=True).wait()
            time.sleep(2)
            subprocess.Popen('mpirun -np 32 ./a.out', shell=True).wait()
            os.rename('sq.txt', 'sq_{}.txt'.format(fname.split('.')[-1]))

def group2blob(binary_2D):

    h, w = binary_2D.shape

    blobs = []
    flag=0
    while (flag == 0):
        candidate=[]
        blob=[]
        for i in range(h):
            for j in range(w):
                if binary_2D[i,j] == 1:
                    candidate.append([i,j])
        if len(candidate) > 0:
            floodfill_pbc(binary_2D, candidate[0][0], candidate[0][1], blob)
            blobs.append(blob)
        else:
            flag=1

    return blobs

def floodfill_pbc(binary_2D, x, y, blob, repeat=4):
    """
    Floodfill over periodic boundary conditions
    Arguments:
    ----------
    binary_2D: A 2D array (binx by biny) whose elements represent atoms counts or weighted intensities
    x, y: indeces for binary_2D[x][y]
    blob: sets of [i, j] for each object determined by floodfill
    repeat: floodfill is applied over this many periods
    Returns:
    --------
    """
    array = binary_2D

    if array[x%len(array), y%len(array[x%len(array)])] == 1:
        array[x%len(array), y%len(array[x%len(array)])] = 0.5
        blob.append([x,y])

        if x > len(array)*(-repeat):
            floodfill_pbc(array, x-1, y, blob)
        if x < repeat*len(array)-1:
            floodfill_pbc(array, x+1, y, blob)
        if y > len(array[x%len(array)])*(-repeat):
            floodfill_pbc(array, x, y-1, blob)
        if y < repeat*len(array[x%len(array)])-1:
            floodfill_pbc(array, x, y+1, blob)
