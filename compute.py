import numpy as np
import sys
import glob
import os
import re
import subprocess
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

try:
    from mpi4py import MPI
except:
    print("no MPI")

try:
    plt.rcParams["text.usetex"] = True
except:
    print("no tex")

class dumpobj():

    def __init__(self, dirname_pattern, dir_end, fname_pattern, Nevery, Nrepeat, Nfreq, end):

        self.dirname_pattern = dirname_pattern
        self.dir_end = dir_end
        self.fname_pattern = fname_pattern
        self.Nevery = Nevery
        self.Nrepeat = Nrepeat
        self.Nfreq = Nfreq
        self.end = end

        self.source_dirs = []
        for iind in range(dir_end + 1):
            self.source_dirs.append(dirname_pattern.format(iind))

        self.freqs = [i*Nfreq for i in (np.arange(int(end/Nfreq)) + 1)]

        fnames = []
        fnames.append(glob.glob(os.path.join(self.source_dirs[0], self.fname_pattern.format(0)))[0])
        for dir_ind, source_dir in enumerate(self.source_dirs):

            for freq_ind, freq in enumerate(self.freqs):
                patterns = [fname_pattern.format(i) for i in np.linspace(freq - self.Nevery*(self.Nrepeat -1), freq, self.Nrepeat, dtype=int)]

                for pattern_ind, pattern in enumerate(patterns):
                    fnames.append(glob.glob(os.path.join(source_dir, pattern))[0])
        
        self.fnames = fnames

    def run(self, protocols, comm=None):
        # TODO: run several functions on the trj at the same time

        fnames = []
        for dir_ind, source_dir in enumerate(self.source_dirs):

            for freq_ind, freq in enumerate(self.freqs):
                patterns = [fname_pattern.format(i) for i in np.linspace(freq - self.Nevery*(self.Nrepeat -1), freq, self.Nrepeat, dtype=int)]

                for pattern_ind, pattern in enumerate(patterns):
                    fnames.append(glob.glob(os.path.join(source_dir, pattern))[0])

        res = {}
        #if 'density' in protocol:
        #    res.update({
        #        'density':
        #        })

        if comm is not None:
            size = comm.Get_size()
            rank = comm.Get_rank()
        
            avg_rows_per_process = int(len(fnames)/size)
        
            start_row = rank * avg_rows_per_process
            end_row = start_row + avg_rows_per_process
            if rank == size - 1:
                end_row = len(fnames)
    
            #if rank == 0:

            density_tmp = np.empty([len(fnames), self.n_bins - 1])

            for iind in range(start_row, end_row):
                fname = fnames[iind]
                trj = np.loadtxt(fname, skiprows=9)
                trj = trj[np.argsort(trj[:,0])]

                #if protocols is not None:
                    #for protocol in protocols: 

                #density_tmp[iind] = c_density()  

    def density(self, lx, ly, zlo, zhi, n_bins, comm):
        self.lx = lx
        self.ly = ly
        self.zlo = zlo
        self.zhi = zhi
        self.n_bins = n_bins
        self.bins = np.linspace(zlo, zhi, n_bins)

        #comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

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

            density_tmp[iind, :] = self.compute_density(trj)

            del trj

        if rank == 0:

            for iind in range(1, size):
                start_row = iind*avg_rows_per_process
                end_row = start_row + avg_rows_per_process
                if iind == size - 1:
                    end_row = len(self.fnames)

                recv = np.empty([len(self.fnames), self.n_bins - 1])
                req = comm.Irecv(recv, source=iind)
                req.Wait()

                density_tmp[start_row:end_row] = recv[start_row:end_row]
        else:
            send = density_tmp
            req = comm.Isend(send, dest=0)
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
            
            cmaps = {}
            from matplotlib.colors import ListedColormap
            W2B = np.dstack((np.linspace(1,0,256), np.linspace(1,0,256), np.linspace(1,0,256)))
            cmaps['W2B'] = ListedColormap(W2B[0], name='W2B')

            fig, ax = plt.subplots(figsize=(4, 3), dpi=1000)
            #plt.rc('font',size=9)

            im = ax.imshow(density_final, vmin=0, vmax=0.5, cmap=cmaps['W2B'], origin='lower')
            ax.set_aspect('auto')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_ticks([0, 0.5])
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

            return density_final

    def compute_density(self, trj, **args):
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

    def local_fC(self, lx, ly, zlo, zhi, n_bins, comm):
        self.lx = lx
        self.ly = ly
        self.zlo = zlo
        self.zhi = zhi
        self.n_bins = n_bins
        self.bins = np.linspace(zlo, zhi, n_bins)

        #comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

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

            density_tmp[iind, :] = self.compute_fC(trj)

            del trj

        if rank == 0:

            for iind in range(1, size):
                start_row = iind*avg_rows_per_process
                end_row = start_row + avg_rows_per_process
                if iind == size - 1:
                    end_row = len(self.fnames)

                recv = np.empty([len(self.fnames), self.n_bins - 1])
                req = comm.Irecv(recv, source=iind)
                req.Wait()

                density_tmp[start_row:end_row] = recv[start_row:end_row]
        else:
            send = density_tmp
            req = comm.Isend(send, dest=0)
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
            
            cmaps = {}
            from matplotlib.colors import ListedColormap
            W2B = np.dstack((np.linspace(1,0,256), np.linspace(1,0,256), np.linspace(1,0,256)))
            cmaps['W2B'] = ListedColormap(W2B[0], name='W2B')
            G2B = np.dstack((np.linspace(0,1,256), np.linspace(100/255,1,256), np.linspace(0,0,256)))
            cmaps['G2B'] = ListedColormap(G2B[0], name='G2B')

            fig, ax = plt.subplots(figsize=(4, 3), dpi=1000)
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

    def compute_fC(self, trj, **args):
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

    def orientation(self, lx, ly, zlo, zhi, n_bins, w, director, comm):
        self.lx = lx
        self.ly = ly
        self.zlo = zlo
        self.zhi = zhi
        self.n_bins = n_bins
        self.w = w
        self.director = np.asarray(director)
        #self.bins = np.linspace(zlo, zhi, n_bins)

        #comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

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
            bonds = bonds[(bonds[:,1] == 3) | (bonds[:,1] == 6)].astype(int)
            self.bonds = bonds

        else:
            bonds = None

        bonds = comm.bcast(bonds, root=0)
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

            cos_tmp = self.compute_cos(trj)
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
                req = comm.Irecv(recv, source=iind)
                req.Wait()

                S_1D[start_row:end_row, :] = recv[start_row:end_row]
        else:
            send = S_tmp
            req = comm.Isend(send, dest=0)
            req.Wait()

        if rank == 0:
            S_initial = S_1D[0]
            S_all = S_1D[1:].reshape(len(self.source_dirs), len(self.freqs), self.Nrepeat, n_bins)
            S_avg = np.empty([len(self.source_dirs), len(self.freqs), n_bins])

            for dir_ind, source_dir in enumerate(self.source_dirs):
                for freq_ind, freq in enumerate(self.freqs):
                    S_avg[dir_ind, freq_ind, :] = S_all[dir_ind, freq_ind, :, :].mean(axis=0)

            S_avg = S_avg.reshape(-1, self.n_bins)
            S_final = np.vstack((S_initial, S_avg))
            S_final = S_final.transpose()
            
            cmaps = {}
            from matplotlib.colors import ListedColormap
            R = np.concatenate((np.linspace(0,1,128), np.ones(128)))
            G = np.concatenate((np.linspace(0,1,128), np.linspace(1,0,128)))
            B = np.concatenate((np.ones(128), np.linspace(1,0,128)))
            BWR = np.dstack((R, G, B))
            cmaps['BWR'] = ListedColormap(BWR[0], name='BWR')

            fig, ax = plt.subplots(figsize=(4,3))
            im = ax.imshow(S_final, vmin=-0.2, vmax=0.2, cmap=cmaps['BWR'], origin='lower')
            ax.set_aspect('auto')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_ticks([-0.2, 0, 0.2])
            cbar.set_label(r'$S$')
            xticks = np.linspace(0,len(self.source_dirs)*len(self.freqs),4)
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

    def compute_cos(self, trj):
        
        A = trj[self.bonds[:,2] - 1]
        B = trj[self.bonds[:,3] - 1]
        
        #logic = np.logical_and(np.logical_and(A[:,4] > 5, B[:,4] > 5),(np.logical_and(A[:,4] < 25, B[:,4] < 25))) #
        #A = A[logic] #
        #B = B[logic] #

        d = self.d_pbc(A, B)
        z_avg = (A[:,5] + B[:,5])/2

        cos = np.empty([len(self.bonds), 2])
        #cos = np.empty([A.shape[0], 2]) #

        cos[:,0] = z_avg
        cos[:,1] = np.dot(d, self.director) / np.linalg.norm(d, axis=1)

        return cos

    def compute_S(self, cos):

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

    def orientation_plane(self, lx, ly, zlo, zhi, n_bins, w, director, comm):
        self.lx = lx
        self.ly = ly
        self.zlo = zlo
        self.zhi = zhi
        self.n_bins = n_bins
        self.w = w
        self.director = np.asarray(director)
        #self.bins = np.linspace(zlo, zhi, n_bins)

        #comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        if rank == 0:
            #number of bonds varies
            subprocess.Popen('grep -A 182401 "Bonds" equil_0/data_preequil > temp.txt', shell=True).wait()
            bonds = np.loadtxt('temp.txt', skiprows=2)
            subprocess.Popen('rm temp.txt', shell=True).wait()
            bonds = bonds[(bonds[:,1] == 3) | (bonds[:,1] == 6)].astype(int)
            self.bonds = bonds

        else:
            bonds = None

        bonds = comm.bcast(bonds, root=0)
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

    def segregation(self, blend, zlo, zhi, nx, ny, nz, w, comm):
        self.blend = blend
        self.zlo = zlo
        self.zhi = zhi
        self.bins = [nx, ny, nz]
        self.w = w
        #self.bins = np.linspace(zlo, zhi, n_bins)

        #comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        if rank == 0:
            infiles = glob.glob(os.path.join(self.source_dirs[0], self.fname_pattern.format(0)))
            infiles.sort()

            fname = infiles[0]
            #print(fname)

            f = open(fname,'r')
            box = []
            for i in range(5):
                f.readline()
            for i in range(3):
                line = f.readline().split()
                box.append([float(line[0]), float(line[1])])
            f.close()
            self.box = box

            binary_3D_initial, delta_0 = self.binarize_3D_n(fname)

            seg_initial = self.compute_segregation(binary_3D_initial)

        else:
            box = None

        box = comm.bcast(box, root=0)
        self.box = box

        fnames = []
        for dir_ind, source_dir in enumerate(self.source_dirs):

            for freq_ind, freq in enumerate(self.freqs):
                patterns = [self.fname_pattern.format(i) for i in np.linspace(freq - self.Nevery*(self.Nrepeat -1), freq, self.Nrepeat, dtype=int)]

                for pattern_ind, pattern in enumerate(patterns):
                    fnames.append(glob.glob(os.path.join(source_dir, pattern))[0])

        avg_rows_per_process = int(len(fnames)/size)
        
        #print(avg_rows_per_process)

        start_row = rank * avg_rows_per_process
        end_row = start_row + avg_rows_per_process
        if rank == size - 1:
            end_row = len(fnames)

        binary_3D =  {
                'A': np.empty([ny, nx, nz]),
                'B': np.empty([ny, nx, nz])
                }

        seg_tmp = np.empty([len(fnames), nz])

        for iind in range(start_row, end_row):
            tmp, delta = self.binarize_3D_n(fnames[iind])
            binary_3D['A'] = tmp['A']
            binary_3D['B'] = tmp['B']

            tmp = self.compute_segregation(binary_3D)
            seg_tmp[iind, :] = tmp

        if rank == 0:
            seg_1D = np.empty([len(fnames), nz])
            seg_1D[start_row:end_row, :] = seg_tmp[start_row:end_row]

            for iind in range(1, size):
                start_row = iind*avg_rows_per_process
                end_row = start_row + avg_rows_per_process
                if iind == size - 1:
                    end_row = len(fnames)

                recv = np.empty([len(fnames), nz])
                req = comm.Irecv(recv, source=iind)
                req.Wait()

                seg_1D[start_row:end_row, :] = recv[start_row:end_row]
        else:
            send = seg_tmp
            req = comm.Isend(send, dest=0)
            req.Wait()

        if rank == 0:
            seg_all = seg_1D.reshape(len(self.source_dirs), len(self.freqs), self.Nrepeat, nz)
            seg_avg = np.empty([len(self.source_dirs), len(self.freqs), nz])

            for dir_ind, source_dir in enumerate(self.source_dirs):
                for freq_ind, freq in enumerate(self.freqs):
                    seg_avg[dir_ind, freq_ind, :] = seg_all[dir_ind, freq_ind, :, :].mean(axis=0)

            seg_avg = seg_avg.reshape(-1, nz)
            seg_final = np.vstack((seg_initial, seg_avg))
            seg_final = seg_final.transpose()

            #print(seg_final.shape)

            cmaps = {}
            from matplotlib.colors import ListedColormap
            G2B = np.dstack((np.linspace(128/255,1,256), np.linspace(0,1,256), np.linspace(128/255,0,256)))
            cmaps['G2B'] = ListedColormap(G2B[0], name='G2B')

            fig, ax = plt.subplots(figsize=(4,3))
            im = ax.imshow(seg_final, vmin=0, vmax=0.6, cmap=cmaps['G2B'], origin='lower')
            ax.set_aspect('auto')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_ticks([0, 0.6])
            cbar.set_label(r'standard deviation')
            xticks = np.linspace(0,len(self.source_dirs)*len(self.freqs),4)
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

    def compute_segregation(self, binary_3D):

        seg = binary_3D['A'] / (binary_3D['A'] + binary_3D['B'])
        seg[binary_3D['A'] + binary_3D['B'] == 0] = np.nan
        seg[(binary_3D['A'] == 0) & (binary_3D['B'] != 0)] = 0

        seg_1D = seg.reshape(-1)

        seg_1D = seg_1D[~np.isnan(seg_1D)]

        fA_bins = np.linspace(0, 1, 21)
        hist, bin_edges = np.histogram(seg_1D, bins=fA_bins)
        fA = bin_edges[1:] - (bin_edges[1] - bin_edges[0])/2

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
        delta[0] = (self.box[0][1] - self.box[0][0])/self.bins[0]
        delta[1] = (self.box[1][1] - self.box[1][0])/self.bins[1] 
        delta[2] = self.w

        #print(delta)

        trjs_filter = {}
        if self.blend == 1:
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

    def chisq(self, blend, delta, run_args, comm, **kwargs):

        self.blend = blend
        self.delta = delta

        if 'remain' in run_args:
            if run_args['remain'] == 'A':
                self.phase = [i[0] for i in run_args['phase']]
            else:
                self.phase = [i[1] for i in run_args['phase']]

        #comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

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

        box = comm.bcast(box, root=0)
        dims = comm.bcast(dims, root=0)
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

            topview[iind] = self.view_xy(self.binarize_3D(trj)).reshape(-1)

            del trj

        if rank == 0:

            for iind in range(1, size):
                start_row = iind*avg_rows_per_process
                end_row = start_row + avg_rows_per_process
                if iind == size - 1:
                    end_row = len(self.fnames)

                #recv = np.empty([len(fnames), self.dims['h'], self.dims['w']])
                recv = np.empty([len(self.fnames), self.dims['h'] * self.dims['w']])
                req = comm.Irecv(recv, source=iind)
                req.Wait()

                topview[start_row:end_row] = recv[start_row:end_row]

                del recv

        else:
            #send = np.empty([len(fnames), self.dims['h'], self.dims['w']])
            send = np.empty([len(self.fnames), self.dims['h'] * self.dims['w']])
            for iind in range(start_row, end_row):
                send[iind] = topview[iind]
            req = comm.Isend(send, dest=0)
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

        topview_final = comm.bcast(topview_final, root=0)

        avg_rows_per_process = int(len(topview_final)/size)
        
        start_row = rank * avg_rows_per_process
        end_row = start_row + avg_rows_per_process
        if rank == size - 1:
            end_row = len(topview_final)

        chisq = np.empty(len(topview_final))

        for iind in range(start_row, end_row):
            chisq[iind] = self.compute_chisq(topview_final[iind].reshape(self.dims['h'], self.dims['w']), iind, **run_args)

        if rank == 0:

            for iind in range(1, size):
                start_row = iind*avg_rows_per_process
                end_row = start_row + avg_rows_per_process
                if iind == size - 1:
                    end_row = len(topview_final)

                recv = np.empty(len(topview_final))
                req = comm.Irecv(recv, source=iind)
                req.Wait()

                chisq[start_row:end_row] = recv[start_row:end_row]
        else:
            send = chisq
            req = comm.Isend(send, dest=0)
            req.Wait()

        if rank == 0:
            for iind, val in enumerate(chisq):
                print(iind, val)

            np.savetxt('chisq.txt', chisq)

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

    def binarize_3D(self, trj):

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

        return binary

    def view_xy(self, binary_3D):

        array = binary_3D

        'sum up over thickness'
        topview = np.sum(binary_3D, axis=2)

        'averagve over thickness'
        #topview = np.average(binary_3D, axis=2)

        return topview

    def compute_chisq(self, topview, num, **kwargs):

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

        q = np.linspace(-3.0, 3.0, 7)
        qq_x = fft_resolution//2 + q/q_del
        qq_y = fft_resolution//2 + q/q_del

        fft = np.fft.fft2(topview)
        fshift = np.fft.fftshift(fft)
        magnitude_spectrum_shift = np.abs(fshift)
        #magnitude_spectrum_shift = 20*np.log(np.abs(fshift))
        #magnitude_spectrum_shift = np.log(np.abs(fshift))

        x = np.linspace(0, self.dims['w']-1, self.dims['w'])
        y = np.linspace(0, self.dims['h']-1, self.dims['h'])
        z = magnitude_spectrum_shift

        from scipy import interpolate
        f = interpolate.interp2d(x, y, z, kind='linear')

        xnew = np.linspace(0,self.dims['w']-1,fft_resolution)
        ynew = np.linspace(0,self.dims['h']-1,fft_resolution)
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
        plt.savefig('{}_2Dfft.png'.format(num), dpi=1000)
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

        r *= self.scale # Scale r to convert distance to pixel

        " Fit a gaussian plus line curve to I-q "
        left_ind = 30
        right_ind = 120
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

            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(r*q_del_px, mean, 'o', mec='None', ms=2)
            ax.plot(x*q_del_px, gaussian_plus_line(mi.params, x), 'r-')
            ax.set_xlabel(r'$q$ (1/px)')
            ax.set_ylabel(r'$I$ (a.u.)')
            plt.tight_layout(pad=1, h_pad=None, w_pad=None, rect=None)
            plt.savefig('{}_1Dfft.png'.format(num), dpi=1000)
            #plt.savefig(os.path.join(self.path, 'results', 'Iq.png'), dpi=1000)
            plt.close()

            #self.l0 = l0
            #self.l0_std = l0 - l0_minus_sigma

            return mi.chisqr

        except:
            print("Could not fit a curve")
            self.l0 = 15.0
            self.l0_std = 0.0

    def position(self, comm):

        #moi
        #molnumber, N, fA
        mol_of_interest = np.loadtxt('moi.txt')

        size = comm.Get_size()
        rank = comm.Get_rank()

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
                req = comm.Irecv(recv, source=iind)
                req.Wait()

                position_z[start_row:end_row] = recv[start_row:end_row]
        else:
            send = density_tmp
            req = comm.Isend(send, dest=0)
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

            fig, ax = plt.subplots(figsize=(4, 3), dpi=1000)
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

