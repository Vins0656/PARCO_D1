# PARCO_D1

> **Note:**  
> The `.mtx` files of the chosen matrices or their compressed ZIPs exceed 25 MB.  
> Therefore, the only way to re-run the experiments is to:
> 1. Download this folder.
> 2. Download the `matrices.tar.gz` archives from the following Suite Sparse Matrix Collection links.
> 3. Decompress the archives.
> 4. Add the `.mtx` files into the folder downloaded from this repository.

## Matrix download links

1. [GL7d17](http://sparse.tamu.edu/JGD_GL7d/GL7d17)  
2. [GL7d21](http://sparse.tamu.edu/JGD_GL7d/GL7d21)  
3. [t2em](http://sparse.tamu.edu/CEMW/t2em)  
4. [rail2586](http://sparse.tamu.edu/Mittelmann/rail2586)  
5. [wiki-talk-temporal](http://sparse.tamu.edu/SNAP/wiki-talk-temporal)  
6. [engine](http://sparse.tamu.edu/TKK/engine)  
7. [kron_g500-logn16](http://sparse.tamu.edu/DIMACS10/kron_g500-logn16)  
8. [rel9](http://sparse.tamu.edu/JGD_Relat/rel9)  

## Instructions

### 1) Run the full experiments

- Decompress all the `.tar.gz` files using the tar command, for example:  
  ```bash
  tar -xzf wiki-talk-temporal.tar.gz
  tar -xzf GL7d17.tar.gz
  tar -xzf GL7d21.tar.gz
  # ... repeat for other matrices
  tar -xzf rel9.tar.gz
  ```
- Find the `<matrix_name>.mtx` files and copy them into the folder you downloaded from this repository.
- Upload the folder to the cluster and run:  
  ```bash
  qsub submit_close.pbs
  ```
  or  
  ```bash
  qsub submit_spread.pbs
  ```
  depending on the binding scheme you want to use.
- After the test completes, two output files will be generated:  
  - `risultati_1_close.txt` and `risultati_2_close.txt` if run in close mode.  
  - `risultati_1_spread.txt` and `risultati_2_spread.txt` if run in spread mode.

### 2) Plot strong scaling and efficiency

- Run the plotting script with one of the result files as parameter to generate PNG plots:  
  ```bash
  python plot_scaling.py risultati_1_close.txt
  ```
- The script will generate two images containing the strong scaling and efficiency plots for the given data.

### 3) Run profiling tests

- On the cluster, run one of the profiling jobs by submitting the respective PBS file, for example:  
  ```bash
  qsub submit_perf_albus_close.pbs
  ```
- Upon completion, a folder named like `perf_out_*_*/parcoD1_*_*` will be created, containing performance analysis results of that kernel on all matrix datasets with the specified binding mode. The pbs files are : submit_perf_albus_close.pbs, submit_perf_albus_spread.pbs, submit_perf_auto, submit_perf_dynamic, submit_perf_guided, submit_perf_static
