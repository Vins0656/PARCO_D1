
using namespace std;
#include "parcoD1.h"
#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>



int row_dim, col_dim, num_nnz;

int main(int argc, char* argv[]){
    vector<int> row_ptr, csr_col, csr_val, rand_vector;
    if (!LoadCSR(argc, argv[1], row_dim, col_dim, num_nnz, row_ptr, csr_col, csr_val)) {
        std::cerr << "Error loading CSR data." << endl;
        return -1;
    }
    

    long double duration_omp=0;
    RandVecInit(rand_vector, col_dim, 1, 100);

    int NUM_THREADS=0;

    #pragma omp parallel
    {
        #pragma omp master
        {
            NUM_THREADS = omp_get_num_threads();
        }
    }
    int IT_multiplier = sqrt(NUM_THREADS)+1;
    int adjusted_NUM_IT = NUM_IT * IT_multiplier;

    
    
    
    vector<long long> result_omp, result_albus;
    result_omp.resize(row_dim);
    result_albus.resize(row_dim);
    

    double mean_omp=0;
   
    // SPMV static
    //Warm-up
    for(int i=0; i < 3 ; i++){
        spmvOMP_static(row_ptr, csr_col, csr_val, rand_vector, result_omp);
    }

    fill(result_omp.begin(), result_omp.end(), 0);
    duration_omp=0;

    for(int i=0; i < adjusted_NUM_IT ; i++){
        duration_omp +=1/spmvOMP_static(row_ptr, csr_col, csr_val, rand_vector, result_omp);
    }

    mean_omp =  adjusted_NUM_IT / duration_omp;
    cout << "Average time OMP static (tot_dur/: "<< adjusted_NUM_IT << "): " << mean_omp << " µs" << endl;
    duration_omp=0;


    // SPMV DYNAMIC
    for(int i=0; i < 3 ; i++){
        spmvOMP_dynamic(row_ptr, csr_col, csr_val, rand_vector, result_omp);
    }

    fill(result_omp.begin(), result_omp.end(), 0);

    for(int i=0; i < adjusted_NUM_IT ; i++){
        duration_omp+=1/spmvOMP_dynamic(row_ptr, csr_col, csr_val, rand_vector, result_omp);
    }

    mean_omp =  adjusted_NUM_IT / duration_omp;
    cout << "Average time OMP dynamic (tot_dur/: "<< adjusted_NUM_IT << "): " << mean_omp << " µs" << endl;
    duration_omp=0;

    // SPMV GUIDED
    for(int i=0; i < 3 ; i++){
        spmvOMP_guided(row_ptr, csr_col, csr_val, rand_vector, result_omp);
    }
    
    fill(result_omp.begin(), result_omp.end(), 0);

    for(int i=0; i < adjusted_NUM_IT ; i++){
        duration_omp += 1/spmvOMP_guided(row_ptr, csr_col, csr_val, rand_vector, result_omp);
    }

    mean_omp =  adjusted_NUM_IT / duration_omp;
    cout << "Average time OMP guided ( tot_dur/: "<< adjusted_NUM_IT << "): " << mean_omp << " µs" << endl;
    duration_omp=0;


    // SPMV AUTO
    for(int i=0; i < 3 ; i++){
        spmvOMP_auto(row_ptr, csr_col, csr_val, rand_vector, result_omp);
    }

    fill(result_omp.begin(), result_omp.end(), 0);

    for(int i=0; i < adjusted_NUM_IT ; i++){
        duration_omp+=1/spmvOMP_auto(row_ptr, csr_col, csr_val, rand_vector, result_omp);
    }

    mean_omp =  adjusted_NUM_IT / duration_omp;
    cout << "Average time OMP auto (tot_dur/: "<< adjusted_NUM_IT << "): " << mean_omp << " µs" << endl;
    
    
    // SPMV ALBUS
    long double duration_albus=0;
    for(int i=0; i < 3 ; i++){
        spmvALBUS(row_ptr, csr_col, csr_val, rand_vector, result_albus);
    }
    fill(result_albus.begin(), result_albus.end(), 0);
    for(int i=0; i < adjusted_NUM_IT ; i++){
        duration_albus+= 1/spmvALBUS(row_ptr, csr_col, csr_val, rand_vector, result_albus);
    }

    double mean_albus =  adjusted_NUM_IT / duration_albus ;
    cout << "Average time ALBUS (tot_dur/: "<< adjusted_NUM_IT << "): " << mean_albus << " µs" << endl; 
    bool error_check = false;
    for(int i=0; i< result_omp.size(); i++){
        if(result_omp[i] != result_albus[i]){
            error_check = true;
            cout << "Error at index " << i << ": OMP " << result_omp[i] << " ALBUS " << result_albus[i] << endl;
            break;
        }
    }
    
    if(error_check){
        cout << "ATTENTION: ALBUS result does not match OMP result." << endl;
    }
    return 0;
}
