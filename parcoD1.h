using namespace std;
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

#ifndef PARCOD1
#define PARCOD1



const int NUM_IT = 100;
extern int row_dim, col_dim, num_nnz;



void RandVecInit(vector<int>& rand_vector, int size, int min_val, int max_val);

bool LoadCSR(int argc, char* filename, int& row_dim, int& col_dim, int& num_nnz, 
              vector<int>& row_ptr, vector<int>& csr_col, 
              vector<int>& csr_val);

double spmvOMP_static(const vector<int>& row_ptr,
                     const vector<int>& csr_col,
                     const vector<int>& csr_val,
                     const vector<int>& rand_vector,
                     vector<long long>& result
                     );

double spmvOMP_auto(const vector<int>& row_ptr,
const vector<int>& csr_col,
const vector<int>& csr_val,
const vector<int>& rand_vector,
vector<long long>& result);

double spmvOMP_guided(const vector<int>& row_ptr,
const vector<int>& csr_col,
const vector<int>& csr_val,
const vector<int>& rand_vector,
vector<long long>& result
);

double spmvOMP_dynamic(const vector<int>& row_ptr,
const vector<int>& csr_col,
const vector<int>& csr_val,
const vector<int>& rand_vector,
vector<long long>& result);

double spmvALBUS(const std::vector<int>& row_ptr,
               const std::vector<int>& csr_col,
               const std::vector<int>& csr_val,
               const std::vector<int>& x,
               std::vector<long long>& y);
 
double Moltiplicazione(const vector<long long>& row_ptr,
                     const vector<long long>& csr_col,
                     const vector<long long>& csr_val,
                     const vector<long long>& rand_vector);               


#endif