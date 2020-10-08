## Test System

GaTech PACE-Hive cluster GPU partition

2 * Xeon Gold 6226, Mellanox ConnectX-5 EDR IB

RHEL 7.6, ICC 19.0.5

[GDRCopy](https://github.com/NVIDIA/gdrcopy) and [nv_peer_mem](https://github.com/Mellanox/nv_peer_memory) not installed yet

## MVAPICH2 2.3.4 

Note: self-compiled, not GDR version

`sendrecv` and `send_recv` : crashes when running on multiple nodes and message size >= 4325 * 4 bytes.

`put_and_acc`: single node, program hangs when `MPI_Put` message size >= 1593 * 4 bytes. And the program crashes when calling `MPI_Accumulate`. 

## OpenMPI 3.1.6

`sendrecv` and `send_recv` : single node, program crashes. This problem does not exist in OpenMPI 3.0.6.

## MPICH 3.4b1

ch3: single node, `sendrecv` and `send_recv` program crashes

ch4+ucx: single node, `sendrecv` and `send_recv` program crashes

ch4+ofi: even the mpi_helloworld crashes