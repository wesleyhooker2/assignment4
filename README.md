# assignment4

g++ -mavx512f -I/opt/kokkos-openmp/include -I/opt/openmpi-4.1.0/include -fopenmp -O3 -c hw4.cc -o hw4.o
g++ -mavx512f -fopenmp -L/opt/kokkos-openmp/lib -L/opt/openmpi-4.1.0/lib -Wl,-rpath -Wl,/opt/openmpi-4.1.0/lib hw4.o -lkokkoscore -ldl -lmpi -o hw4.host
./hw4.host
/opt/openmpi-4.1.0/bin/mpiexec -n 4 hw4.host

g++ -mavx512f -I/opt/kokkos-openmp/include -I/opt/openmpi-4.1.0/include -fopenmp -O3 -c hw4.cc -o hw4.o &&
g++ -mavx512f -fopenmp -L/opt/kokkos-openmp/lib -L/opt/openmpi-4.1.0/lib -Wl,-rpath -Wl,/opt/openmpi-4.1.0/lib hw4.o -lkokkoscore -ldl -lmpi -o hw4.host &&
./hw4.host

g++ -mavx512f -I/opt/kokkos-openmp/include -I/opt/openmpi-4.1.0/include -fopenmp -O3 -c hw4.cc -o hw4.o &&
g++ -mavx512f -fopenmp -L/opt/kokkos-openmp/lib -L/opt/openmpi-4.1.0/lib -Wl,-rpath -Wl,/opt/openmpi-4.1.0/lib hw4.o -lkokkoscore -ldl -lmpi -o hw4.host &&
/opt/openmpi-4.1.0/bin/mpiexec -n 4 hw4.host
