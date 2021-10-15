import compute

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
except:
    print('no mpi4py')

#path = '/data3/N2022_f2550/C20n7250onL22n6591_0.00003Mtau'
path = None
dumps = compute.dumpobj(path=path, dir_start=0, dir_end=0,
        Nevery=10000, Nrepeat=1, Nfreq=10000, end=10000,
        comm=comm)

#a = compute.dumpobj(path=path, dirname_pattern='equil_{}', dir_end=0, fname_pattern='dump*.{}', Nevery=10000, Nrepeat=1, Nfreq=10000, end=10000000)
#a = compute.dumpobj(dirname_pattern='equil_{}_quench_0', dir_end=0, fname_pattern='dump*.{}', Nevery=10000, Nrepeat=1, Nfreq=10000000, end=10000000)

#a = compute.dumpobj(path=path, dirname_pattern='equil_{}', dir_end=0, fname_pattern='dump*.{}', Nevery=10000, Nrepeat=1, Nfreq=10000, end=100000)
#a.sf([1, 3], 31)

#density = a.local_fC(57, 90, 0, 75, 101, comm)
#density = a.density(57, 90, 0, 40, 61, comm)
#density = a.density(57, 90, 0, 70, 141, comm)
#S = a.orientation(57, 90, 0, 50, 100, 0.5, [0, 0, 1], comm)
#S = a.orientation(57, 90, 0, 70, 100, 2.0, [0, 0, 1], comm)
dumps.orientation(57, 90, 0, 70, 140, 0.5, [0, 0, 1])
dumps.segregation(1, 0, 70, 14, 22, 141, 0.5)
#concentration = a.concentration(57, 90, 70, 'L', comm)
#concentration = a.concentration(57, 90, 48, 'L', comm)

#if rank == 0:
#    print(density.shape)
#    print(S.shape)
#    print(Seg.shape)

run_args = {
        #'phase': [[1, 2], [3, 4]],
        'phase': [[1, 2],],
        'remain': 'A',
        'threshold': 400,
        'distance_in_pixels': 2,
        'known_distance': 1,
        'distance_unit': 'sigma',
        'fft_resolution': 2048, # pixels
        'zoom_mag': 4,
        }

#a.chisq(1, [0.5, 0.5, 0.5], run_args, comm)
