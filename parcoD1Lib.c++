using namespace std;
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <omp.h>
#include "parcoD1.h"

void RandVecInit(vector<int>& rand_vector, int size, int min_val, int max_val) {
    random_device rd;                         // seme casuale da hardware
    mt19937 gen(rd());                        // generatore Mersenne Twister
    uniform_int_distribution<> dis(min_val, max_val); // distribuzione uniforme tra min_val e max_val
    rand_vector.resize(size);
    for (int i = 0; i < size; i++) {
        rand_vector[i] = dis(gen);
        // cout << "rand_vector[" << i << "]: " << rand_vector[i] << endl;
    }
}


bool LoadCSR(int argc, char* filename, int& row_dim, int& col_dim, int& num_nnz, 
              vector<int>& row_ptr, vector<int>& csr_col, 
              vector<int>& csr_val){
    if (argc<2){
        cout << "missing <input_file>" << endl;
        return false;
    }
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Error opening file: " << filename << endl;
        return false;
    }

    string line;
    // Read matrix dimensions
     while(getline(file,line)) {
        if(line[0] != '%') break; 
    }

    stringstream ss(line);
    ss >> row_dim >> col_dim >> num_nnz;
    cout << row_dim << " " << col_dim << " " << num_nnz << endl;
    
  
    vector<int> row_ind(num_nnz);
    vector<int> col_ind(num_nnz);
    vector<int> val(num_nnz);
    
    for(int i=0; i < num_nnz; i++) { 
        file >> row_ind[i] >> col_ind[i] >> val[i]; 
        row_ind[i]--; // 1-based a 0-based
        col_ind[i]--; // 1-based a 0-based
    }
    file.close();

    // --- Conversione a CSR ---

    row_ptr.assign(row_dim + 1, 0);
    csr_col.resize(num_nnz);
    csr_val.resize(num_nnz);

    // 1. Costruisci l'istogramma (Pass 1)
    for(int i=0; i < num_nnz; i++) {
        // --- CONTROLLO 1 ---
        // Ignora gli indici non validi (es. -1)
        if (row_ind[i] >= 0 && row_ind[i] < row_dim) { 
            row_ptr[row_ind[i] + 1]++;
        }
    }

    // 2. Calcola la prefix-sum
    for(int i=1; i <= row_dim; i++) {
        row_ptr[i] += row_ptr[i-1]; 
    }

    
    csr_col.resize(num_nnz);
    csr_val.resize(num_nnz);

    // 3. Inserisci i dati (Pass 2)
    vector<int> temp_row_ptr = row_ptr;
    for(int i=0; i < num_nnz; i++) {
        
        // --- CONTROLLO 2 ---
        int r = row_ind[i];
        int c = col_ind[i];
        
        // Ignora di nuovo gli indici non validi
        if (r >= 0 && r < row_dim && c >= 0 && c < col_dim) {
            int dest = temp_row_ptr[r]++; // <-- Questa era la riga 219
            csr_col[dest] = c;
            csr_val[dest] = val[i];
        }
    }

    num_nnz = row_ptr[row_dim];
    
    return true;
}

double spmvOMP_static(const vector<int>& row_ptr,
            const vector<int>& csr_col,
            const vector<int>& csr_val,
            const vector<int>& rand_vector,
            vector<long long>& result
          )
{
    auto start = std::chrono::high_resolution_clock::now();
    const int nrows = row_ptr.size()-1;
    #pragma omp parallel for schedule (static, 512)
    for(int i =0; i < nrows; i++){
        long long sum=0;
        #pragma omp simd reduction(+:sum)
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            sum += (long long)csr_val[j] * rand_vector[csr_col[j]];
        }
        result[i] = sum;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count();
}   


double spmvOMP_guided(const vector<int>& row_ptr,
            const vector<int>& csr_col,
            const vector<int>& csr_val,
            const vector<int>& rand_vector,
            vector<long long>& result
            )
{
    auto start = std::chrono::high_resolution_clock::now();
    const int nrows = row_ptr.size()-1;
    #pragma omp parallel for schedule (guided,512)
    for(int i =0; i < nrows; i++){
        long long sum=0;
        #pragma omp simd reduction(+:sum)
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            sum += (long long)csr_val[j] * rand_vector[csr_col[j]];
        }
        result[i] = sum;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count();
}         

double spmvOMP_auto(const vector<int>& row_ptr,
            const vector<int>& csr_col,
            const vector<int>& csr_val,
            const vector<int>& rand_vector,
            vector<long long>& result)
{
    auto start = std::chrono::high_resolution_clock::now();
    const int nrows = row_ptr.size()-1;
    #pragma omp parallel for schedule (auto)
    for(int i =0; i < nrows; i++){
        long long sum=0;
        #pragma omp simd reduction(+:sum)
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            sum += (long long)csr_val[j] * rand_vector[csr_col[j]];
        }
        result[i] = sum;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count();
}         

double spmvOMP_dynamic(const std::vector<int>& row_ptr,
               const std::vector<int>& csr_col,
               const std::vector<int>& csr_val,
               const std::vector<int>& x,
               std::vector<long long>& y)
{
    auto start = std::chrono::high_resolution_clock::now();
    const int nrows = row_ptr.size()-1;
    #pragma omp parallel for schedule (dynamic, 1024)
    for(int i =0; i < nrows; i++){
        long long sum=0;
        #pragma omp simd reduction(+:sum)
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            sum += (long long)csr_val[j] * x[csr_col[j]];
        }
        y[i] = sum;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count();
}


double spmvALBUS(const std::vector<int>& row_ptr,
               const std::vector<int>& csr_col,
               const std::vector<int>& csr_val,
               const std::vector<int>& x,
               std::vector<long long>& y)
{
    const int row_dim = static_cast<int>(row_ptr.size()) - 1;
    const int nnz     = row_ptr[row_dim];
    y.assign(row_dim, 0LL);

    // Numero di thread effettivi (da runtime OMP)
    int T = 1;
    #pragma omp parallel
    {
        #pragma omp master
        { T = omp_get_num_threads(); }
    }
    if (T < 1) T = 1;

    // lower_bound sul prefisso CSR: primo r con row_ptr[r] >= cut
    auto row_from_cut = [&](int cut) -> int {
        int L = 0, R = row_dim;
        while (L <= R) {
            int H = (L + R) >> 1;
            if (row_ptr[H] >= cut) R = H - 1;
            else                   L = H + 1;
        }
        return L; // primo >= cut
    };

    // Confini per-thread in NNZ e loro mappatura in (riga, offset)
    std::vector<int> row_lo(T), row_hi(T), off_lo(T), off_hi(T);
    std::vector<int> cut_lo(T), cut_hi(T);
    for (int t = 0; t < T; ++t) {
        const int c0 = static_cast<int>((1LL * nnz * t) / T);
        const int c1 = static_cast<int>((1LL * nnz * (t + 1)) / T);
        cut_lo[t] = c0; cut_hi[t] = c1;

        const int r0 = row_from_cut(c0);
        const int r1 = row_from_cut(c1);
        const int o0 = c0 - (r0 ? row_ptr[r0 - 1] : 0);
        const int o1 = c1 - (r1 ? row_ptr[r1 - 1] : 0);

        row_lo[t] = r0; off_lo[t] = o0;
        row_hi[t] = r1; off_hi[t] = o1;
    }

    // Parziali dei bordi + caso "tutto in una riga"
    std::vector<long long> left_partial(T, 0), right_partial(T, 0), single_partial(T, 0);
    std::vector<int>       single_row(T, -1);

    auto start = std::chrono::high_resolution_clock::now();

    // Kernel parallelo per blocco NNZ
    #pragma omp parallel
    {
        const int t  = omp_get_thread_num();
        const int r0 = row_lo[t], r1 = row_hi[t];
        const int o0 = off_lo[t],  o1 = off_hi[t];
        const int c0 = cut_lo[t],  c1 = cut_hi[t];

        if (c0 == c1) {
            // blocco vuoto
        }
        else if (r0 == r1) {
            // Tutto in una sola riga: segmento [o0, o1)
            const int i  = r0;
            const int js = row_ptr[i] + o0;
            const int je = row_ptr[i] + o1; // esclusivo
            long long acc = 0;
            for (int j = js; j < je; ++j)
                acc += 1LL * csr_val[j] * x[csr_col[j]];
            single_partial[t] = acc;
            single_row[t]     = i;
        }
        else {
            // Riga iniziale: se o0>0 è la coda della riga (r0-1); se o0==0 la riga r0 è piena
            if (o0 > 0) {
                const int i  = r0 - 1;
                const int js = row_ptr[i] + o0;
                const int je = row_ptr[i + 1];
                long long acc = 0;
                for (int j = js; j < je; ++j)
                    acc += 1LL * csr_val[j] * x[csr_col[j]];
                right_partial[t] = acc;
            } else {
                long long sum = 0;
                for (int j = row_ptr[r0]; j < row_ptr[r0 + 1]; ++j)
                    sum += 1LL * csr_val[j] * x[csr_col[j]];
                y[r0] = sum;
            }

            // Righe interne piene:
            // - se o1>0, fino a r1-2; se o1==0, include r1-1 (riga piena)
            const int first_full = (o0 > 0 ? r0 : r0 + 1);
            const int last_full  = (o1 > 0 ? r1 - 2 : r1 - 1);
            
            for (int i = first_full; i <= last_full; ++i) {
                if (i < 0 || i >= row_dim) continue;
                long long sum = 0;
                #pragma omp simd reduction(+:sum)
                for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
                    sum += 1LL * csr_val[j] * x[csr_col[j]];
                y[i] = sum;
            }

            // Riga finale: testa della riga (r1-1) se o1>0
            if (o1 > 0) {
                const int i  = r1 - 1;
                const int js = row_ptr[i];
                const int je = row_ptr[i] + o1; // esclusivo
                long long acc = 0;
                for (int j = js; j < je; ++j)
                    acc += 1LL * csr_val[j] * x[csr_col[j]];
                left_partial[t] = acc;
            }
        }
    }

    // Gather sequenziale dei margini per evitare atomiche
    for (int t = 0; t < T; ++t) {
        const int r0 = row_lo[t], r1 = row_hi[t];
        const int o0 = off_lo[t],  o1 = off_hi[t];

        if (single_row[t] >= 0) {
            y[single_row[t]] += single_partial[t];
        } else if (cut_lo[t] != cut_hi[t]) {
            if (o0 > 0 && (r0 - 1) >= 0)
                y[r0 - 1] += right_partial[t];
            if (o1 > 0 && (r1 - 1) >= 0)
                y[r1 - 1] += left_partial[t];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count();
}


double Moltiplicazione(const vector<long long>& row_ptr,
                     const vector<long long>& csr_col,
                     const vector<long long>& csr_val,
                     const vector<long long>& rand_vector) {

    vector<long long> result(row_ptr.size() - 1, 0); // risultato per ogni riga
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < row_ptr.size() - 1; i++) {
        long long sum = 0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            sum += csr_val[j] * rand_vector[csr_col[j]];
        }
        result[i]=sum;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Tempo di esecuzione: " << duration.count() << " microsecondi" << std::endl;
    return duration.count();
}