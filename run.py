import Compute

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
except:
    print('no mpi4py')

#path = '/data2/N2020_f2525_m5050/both_disordered/alpha_1/Gamma_0.4/s_1'
path = None
dumps = Compute.Dumpobj(path=path, dir_start=25, dir_end=25,
        Nevery=10000, Nrepeat=5, Nfreq=10000000, end=10000000,
        comm=comm)

#import numpy as np
#timesteps = [*np.linspace(0, 1000, 11),
#        *np.linspace(2000, 10000, 9),
#        *np.linspace(20000, 100000, 9),
#        *np.linspace(200000, 1000000, 9)]
#dumps.specify_timesteps(timesteps)

#dumps = Compute.dumpobj(path=path, dir_start=0, dir_end=0,
#        Nevery=10000, Nrepeat=1, Nfreq=10000, end=10000000,
#        comm=comm)

#a = Compute.dumpobj(path=path, dirname_pattern='equil_{}', dir_end=0, fname_pattern='dump*.{}', Nevery=10000, Nrepeat=1, Nfreq=10000, end=10000000)
#a = Compute.dumpobj(dirname_pattern='equil_{}_quench_0', dir_end=0, fname_pattern='dump*.{}', Nevery=10000, Nrepeat=1, Nfreq=10000000, end=10000000)

#a = Compute.dumpobj(path=path, dirname_pattern='equil_{}', dir_end=0, fname_pattern='dump*.{}', Nevery=10000, Nrepeat=1, Nfreq=10000, end=100000)
#a.sf([1, 3], 31)

#density = a.local_fC(57, 90, 0, 75, 101, comm)
#density = a.density(57, 90, 0, 40, 61, comm)
#density = dumps.density(0, 70, 141)
#if rank == 0:
#    import numpy as np
#    np.savetxt('density.txt', density)
#S = a.orientation(57, 90, 0, 50, 100, 0.5, [0, 0, 1], comm)
#S = a.orientation(57, 90, 0, 70, 100, 2.0, [0, 0, 1], comm)
#S = dumps.orientation(0, 70, 140, 0.5, [0, 0, 1])
#if rank == 0:
#    import numpy as np
#    np.savetxt('S.txt', S)
#seg = dumps.segregation(1, 0, 70, 14, 22, 141, 0.5)
#if rank == 0:
#    import numpy as np
#    np.savetxt('seg.txt', seg)
#concentration = dumps.concentration(57, 90, 70, 'L')
#concentration = a.concentration(57, 90, 48, 'L', comm)

#dumps.reconstructFilm(1, [0.5, 0.5, 0.5])
dumps.computeChainAlignment([0, 0, 1])

#dumps.sf([2, 4], 51)

#if rank == 0:
#    print(density.shape)
#    print(S.shape)
#    print(Seg.shape)

run_args = {
        #'phase': [[1, 2], [3, 4]],
        'phase': [[1, 2],], # {1: {'A': 1, 'B': 2}, 2: {'A': 3, 'B': 4}}
        'remain': 'A',
        'threshold': 400,
        'distance_in_pixels': 2,
        'known_distance': 1,
        'distance_unit': 'sigma',
        'fft_resolution': 2048, # pixels
        'zoom_mag': 4,
        }

#a.chisq(1, [0.5, 0.5, 0.5], run_args, comm)

dumps.save_results()
