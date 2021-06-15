import compute
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

a = compute.dumpobj(dirname_pattern='equil_{}', dir_end=0, fname_pattern='dump*.{}', Nevery=10000, Nrepeat=1, Nfreq=10000, end=5000000)
#a = compute.dumpobj(dirname_pattern='equil_{}_quench_0', dir_end=0, fname_pattern='dump*.{}', Nevery=10000, Nrepeat=1, Nfreq=10000000, end=10000000)

#density = a.local_fC(57, 90, 0, 75, 101, comm)
density = a.density(57, 90, 0, 40, 61, comm)
S = a.orientation(57, 90, 0, 40, 60, 2.0, [0, 0, 1], comm)
#Seg = a.segregation(1, 0, 75, 14, 22, 100, 4.0, comm)

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
