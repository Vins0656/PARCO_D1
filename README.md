# PARCO_D1

!! THE .MTX FILES OF THE CHOSEN MATRICES OR THEIR COMPRESSED ZIPS EXCEED 25 MB, THUS THE ONLY WAY TO RE-RUN THE EXPERIMENTS IS DOWNLOAD THIS FOLDER, DOWNLOAD THE MATRICES.TAR.GZ FROM THE FOLLOWING LINKS TO THE SUITE SPARSE MATRIX COLLECTION , DECOMPRESS THE ARCHIVE AND ADD THE .MTX FILE IN THE DOWNLOADED FOLDER FROM THIS REPOSITORY !!

1) http://sparse.tamu.edu/JGD_GL7d/GL7d17
2) http://sparse.tamu.edu/JGD_GL7d/GL7d21
3) http://sparse.tamu.edu/CEMW/t2em
4) http://sparse.tamu.edu/Mittelmann/rail2586
5) http://sparse.tamu.edu/SNAP/wiki-talk-temporal
6) http://sparse.tamu.edu/TKK/engine
7) http://sparse.tamu.edu/DIMACS10/kron_g500-logn16
8) http://sparse.tamu.edu/JGD_Relat/rel9

1) TO RUN THE  FULL EXPERIMENTS
   -Decompress all the matrices .tar.gz files using the tar command: tar -xzf wiki-talk-temporal.tar.gz, tar -xzf GL7d17.tar.gz, tar -xzf GL7d21.tar.gz, ... tar -xzf rel9.tar.gz
   -find the <name of the matrix>.mtx files and copy paste it in the folder downloaded from this repository
   -upload the folder on the cluster and use the qsub submit_close.pbs or submit_spread.pbs to run the version with the desired binding scheme
   -at test completion two files will be generated: risultati_1_close.txt and risultati_2_close.txt if run in close mode or risultati_1_spread.txt and risultati_2_spread.txt if in spread one.
   
2) TO PLOT THE STRONG SCALING AND THE EFFICIENCY
   -run plot_scaling.py  with one of the following files as a parameter to get the plots (strong scaling and efficiency) as a png of those set of matrices:  risultati_1_close.txt / risultati_2_close.txt /   risultati_1_spread.txt / risultati_2_spread.txt
   -at completion the script will generate 2 images containing the plots for the specified program
3)    
   
   
