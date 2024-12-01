# Project Name ITPC_D1

This project implements a matrix transposition algorithm and symmetry check with optimization using OpenMP and AVX. It is designed to test performance on different matrix sizes.

## Prerequisites

- **Language**: C/C++
- **Compiler**: GCC (MinGW) with OpenMP support and AVX2 optimization enabled
- **Libraries**:
  - OpenMP (for parallelization)
  - AVX (Advanced Vector Extensions) for hardware optimization

### libraries included

The project uses some standard libraries and others for performance optimization:

```cpp
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>
#include <vector>
#include <cmath>
#include <immintrin.h>


####compile using

g++ -o ITPC_H1 ITPC_H1_final.c++  -mavx2 -march=native -fopenmp 

#####USAGE

once the execution has started a counter on the screen shows what matrix is abeing elaborated by the program. the code takes a while to get to the end due to the amount of times the functions are re-tested to get statistical accuracy. once its done the results per function and matrix dimension are printed on the terminal.
example of output:

                                           test on matrix dimension: 16x16

 Matrix Transposition                                                           Matrix check
Average time for sequential: 3.15867e-06                                                  Average time for sequential:3.924e-06
Average time for iMP: 7.82667e-07                                                         Average time for iMP: 5.46667e-07
Percentuale di miglioramento seq/IMP:75.2216%                                       Percentuale di miglioramento seq/IMP:86.0686% 
Average time for omp with 1 threads: 0                                          Average time for omp with 1 threads: 0
Percentuale di miglioramento seq/OMP:100%                                       Percentuale di miglioramento seq/OMP:100% 

Average time for omp with 2 threads: 3.08033e-06                                          Average time for omp with 2 threads: 1.719e-06
Percentuale di miglioramento seq/OMP:2.47995%                                       Percentuale di miglioramento seq/OMP:56.1927% 

Average time for omp with 3 threads: 3.27933e-06                                          Average time for omp with 3 threads: 1.70433e-06
Percentuale di miglioramento seq/OMP:-3.82017%                                       Percentuale di miglioramento seq/OMP:56.5664% 

Average time for omp with 4 threads: 3.00067e-06                                          Average time for omp with 4 threads: 1.63533e-06
Percentuale di miglioramento seq/OMP:5.00211%                                       Percentuale di miglioramento seq/OMP:58.3248% 

Average time for omp with 5 threads: 3.12833e-06                                          Average time for omp with 5 threads: 1.708e-06
Percentuale di miglioramento seq/OMP:0.96031%                                       Percentuale di miglioramento seq/OMP:56.473% 

Average time for omp with 6 threads: 3.441e-06                                          Average time for omp with 6 threads: 1.786e-06
Percentuale di miglioramento seq/OMP:-8.93837%                                       Percentuale di miglioramento seq/OMP:54.4852% 

Average time for omp with 7 threads: 3.56867e-06                                          Average time for omp with 7 threads: 1.982e-06
Percentuale di miglioramento seq/OMP:-12.9802%                                       Percentuale di miglioramento seq/OMP:49.4903% 

Average time for omp with 8 threads: 3.95233e-06                                          Average time for omp with 8 threads: 2.34967e-06
Percentuale di miglioramento seq/OMP:-25.1266%                                       Percentuale di miglioramento seq/OMP:40.1206% 


                                            test on matrix dimension: 32x32

 Matrix Transposition                                                           Matrix check
Average time for sequential: 1.63547e-05                                                  Average time for sequential:1.83367e-05
Average time for iMP: 3.45e-06                                                         Average time for iMP: 3.15867e-06
Percentuale di miglioramento seq/IMP:78.9051%                                       Percentuale di miglioramento seq/IMP:82.774%
Average time for omp with 1 threads: 0                                          Average time for omp with 1 threads: 0
Percentuale di miglioramento seq/OMP:100%                                       Percentuale di miglioramento seq/OMP:100%

Average time for omp with 2 threads: 1.55793e-05                                          Average time for omp with 2 threads: 8.47767e-06
Percentuale di miglioramento seq/OMP:4.74074%                                       Percentuale di miglioramento seq/OMP:53.7666%

Average time for omp with 3 threads: 1.8022e-05                                          Average time for omp with 3 threads: 9.23867e-06
Percentuale di miglioramento seq/OMP:-10.1948%                                       Percentuale di miglioramento seq/OMP:49.6164%

Average time for omp with 4 threads: 1.63357e-05                                          Average time for omp with 4 threads: 9.17833e-06
Percentuale di miglioramento seq/OMP:0.11618%                                       Percentuale di miglioramento seq/OMP:49.9455%

Average time for omp with 5 threads: 2.1658e-05                                          Average time for omp with 5 threads: 9.59767e-06
Percentuale di miglioramento seq/OMP:-32.4271%                                       Percentuale di miglioramento seq/OMP:47.6586% 

Average time for omp with 6 threads: 1.70397e-05                                          Average time for omp with 6 threads: 8.894e-06
Percentuale di miglioramento seq/OMP:-4.1884%                                       Percentuale di miglioramento seq/OMP:51.4961%

Average time for omp with 7 threads: 1.63157e-05                                          Average time for omp with 7 threads: 9.34767e-06
Percentuale di miglioramento seq/OMP:0.238471%                                       Percentuale di miglioramento seq/OMP:49.022%

Average time for omp with 8 threads: 1.52547e-05                                          Average time for omp with 8 threads: 8.26467e-06
Percentuale di miglioramento seq/OMP:6.72591%                                       Percentuale di miglioramento seq/OMP:54.9282%


                                            test on matrix dimension: 64x64

 Matrix Transposition                                                           Matrix check
Average time for sequential: 4.20093e-05                                                  Average time for sequential:5.06213e-05
Average time for iMP: 8.63867e-06                                                         Average time for iMP: 5.832e-06
Percentuale di miglioramento seq/IMP:79.4363%                                       Percentuale di miglioramento seq/IMP:88.4792%
Average time for omp with 1 threads: 0                                          Average time for omp with 1 threads: 0
Percentuale di miglioramento seq/OMP:100%                                       Percentuale di miglioramento seq/OMP:100%

Average time for omp with 2 threads: 8.75673e-05                                          Average time for omp with 2 threads: 2.37503e-05
Percentuale di miglioramento seq/OMP:-108.447%                                       Percentuale di miglioramento seq/OMP:53.0824%

Average time for omp with 3 threads: 6.84943e-05                                          Average time for omp with 3 threads: 2.24313e-05
Percentuale di miglioramento seq/OMP:-63.0455%                                       Percentuale di miglioramento seq/OMP:55.688%

Average time for omp with 4 threads: 3.63987e-05                                          Average time for omp with 4 threads: 2.06117e-05
Percentuale di miglioramento seq/OMP:13.3558%                                       Percentuale di miglioramento seq/OMP:59.2826%

Average time for omp with 5 threads: 3.62657e-05                                          Average time for omp with 5 threads: 1.982e-05
Percentuale di miglioramento seq/OMP:13.6724%                                       Percentuale di miglioramento seq/OMP:60.8465%

Average time for omp with 6 threads: 3.6909e-05                                          Average time for omp with 6 threads: 1.9372e-05
Percentuale di miglioramento seq/OMP:12.141%                                       Percentuale di miglioramento seq/OMP:61.7315%

Average time for omp with 7 threads: 3.57073e-05                                          Average time for omp with 7 threads: 2.04563e-05
Percentuale di miglioramento seq/OMP:15.0014%                                       Percentuale di miglioramento seq/OMP:59.5895%

Average time for omp with 8 threads: 3.8533e-05                                          Average time for omp with 8 threads: 2.0281e-05
Percentuale di miglioramento seq/OMP:8.27514%                                       Percentuale di miglioramento seq/OMP:59.9359%



etc. etc.