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

dumps.computeChainAlignment([0, 0, 1])

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
