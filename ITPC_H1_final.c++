//TO DO: aggiungere stampa risultati OMP, MODIFICA WALL CLOCKK ESECUZIONE OMP

using namespace std;
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>
#include <vector>
#include <cmath>
#include <immintrin.h>

typedef std::vector<std::vector<float>> Matrix;

struct timeResult{
    float time_tran;
    float time_check;
};


void init_matrix(int counter, Matrix & mat, bool resize) {
    int dim_array[10] = {8,16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    int dim_n = dim_array[counter];
     
    
    if (resize) {   
    mat.resize(dim_n); 
    for (int i = 0; i < dim_n; i++) {
        mat[i].resize(dim_n);
    }
}


    // Inizializzazione degli elementi della matrice
    for (int row = 0; row < dim_n; row+=1) {
        for (int col = 0; col < dim_n; col+=1) {
            mat[row][col] = float(rand() % 100) / 10;
        }
    }
}

double checkSym(Matrix &mat) {
    int dim = mat.size();
    bool res=true;
    auto start_transpose= chrono::high_resolution_clock::now(); 
    for (int row = 0; row < dim; row++) {
        for (int col = row+1 ; col < dim; col++) {
            if (mat[row][col] != mat[col][row]) {
                res= false;
            }
        }
    }
    auto end_transpose =  chrono::high_resolution_clock::now();
    chrono::duration<double> duration_transpose=end_transpose - start_transpose;

    if (res){
    }
    return duration_transpose.count(); 
}

double checkSymImp(Matrix &matToTran){
    bool check=false;
    int dim= matToTran.size();
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int row = 0; row < dim; row++) {
        for (int col = row + 1; col < dim; col += 8) {
            // Caricamento delle righe e delle colonne nei registri AVX
            __m256 row_values = _mm256_loadu_ps(&matToTran[row][col]);      // Carica 8 valori dalla riga
            __m256 col_values = _mm256_loadu_ps(&matToTran[col][row]);      // Carica 8 valori dalla colonna
            __m256 cmp_result = _mm256_cmp_ps(row_values, col_values, _CMP_EQ_OQ); // Confronto tra i valori

            // Controllo del risultato
            if (_mm256_movemask_ps(cmp_result) != 0xFF) { // Se un confronto non è uguale
                check = false; // Continua a verificare, ma segna la matrice come non simmetrica
            }
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_transpose = end_time - start_time;
    return duration_transpose.count();
}

double checkSymOmp(Matrix &mat, int num_threads) {
    int dim = mat.size();
    bool res = true;  
    float result=0.0;

    omp_set_num_threads(num_threads);

    if (dim <= 64) {
        auto start_transpose= chrono::high_resolution_clock::now(); 
        for (int row = 0; row < dim; row++) {
            #pragma simd
            for (int col = row + 1; col < dim; col++) { 
                if (mat[row][col] != mat[col][row]) {
                        res = false;  
                }
            }
        }
        auto end_transpose =  chrono::high_resolution_clock::now();
        chrono::duration<double> duration_transpose=end_transpose - start_transpose;
        result=duration_transpose.count();
    }
    else 
    {
      auto start_transpose= chrono::high_resolution_clock::now(); 
       #pragma omp parallel for collapse(2) reduction(&:res)
        for (int row = 0; row < dim; row++) {
            for (int col = row + 1; col < dim; col++) { // Only check upper triangle (row, col) pairs where row < col
                // Prefetch data for the next iteration
                _mm_prefetch((const char*)&mat[row][col + 1], _MM_HINT_T0);
                _mm_prefetch((const char*)&mat[col][row + 1], _MM_HINT_T0);

                // Compare the current elements
                if (mat[row][col] != mat[col][row]) {
                    res = false;  // Set to false if a mismatch is found
                }
            }   
        }
        auto end_transpose =  chrono::high_resolution_clock::now();
        chrono::duration<double> duration_transpose=end_transpose - start_transpose;
        result=duration_transpose.count();
    }
    
    return result;
}    


double matTranspose(Matrix &mat, Matrix transposed) {
    int dim = mat.size();
    auto start_transpose = chrono::high_resolution_clock::now();           
    for (int row = 0; row < dim; row++) {
        for (int col = 0; col < dim; col++) {
            transposed[col][row] = mat[row][col];
        }
    }
    auto end_transpose =  chrono::high_resolution_clock::now();
    chrono::duration<double> duration_transpose=end_transpose - start_transpose;
    return duration_transpose.count();
}


double matTransposeImp(Matrix &mat, Matrix transposed) {
    int dim = mat.size(); // Dimensione della matrice (supposta quadrata)
     auto start_transpose = chrono::high_resolution_clock::now();
    // Elaborazione a blocchi di 8x8 per sfruttare l'AVX
    for (int row = 0; row < dim; row += 8) {
        for (int col = 0; col < dim; col += 8) {
            // Gestione dei blocchi 8x8
            __m256 row0 = _mm256_loadu_ps(&mat[row + 0][col]);
            __m256 row1 = _mm256_loadu_ps(&mat[row + 1][col]);
            __m256 row2 = _mm256_loadu_ps(&mat[row + 2][col]);
            __m256 row3 = _mm256_loadu_ps(&mat[row + 3][col]);
            __m256 row4 = _mm256_loadu_ps(&mat[row + 4][col]);
            __m256 row5 = _mm256_loadu_ps(&mat[row + 5][col]);
            __m256 row6 = _mm256_loadu_ps(&mat[row + 6][col]);
            __m256 row7 = _mm256_loadu_ps(&mat[row + 7][col]);

            // Trasposizione del blocco 8x8
            __m256 t0 = _mm256_unpacklo_ps(row0, row1);
            __m256 t1 = _mm256_unpackhi_ps(row0, row1);
            __m256 t2 = _mm256_unpacklo_ps(row2, row3);
            __m256 t3 = _mm256_unpackhi_ps(row2, row3);
            __m256 t4 = _mm256_unpacklo_ps(row4, row5);
            __m256 t5 = _mm256_unpackhi_ps(row4, row5);
            __m256 t6 = _mm256_unpacklo_ps(row6, row7);
            __m256 t7 = _mm256_unpackhi_ps(row6, row7);

            __m256 z0 = _mm256_shuffle_ps(t0, t2, 0x44);
            __m256 z1 = _mm256_shuffle_ps(t0, t2, 0xEE);
            __m256 z2 = _mm256_shuffle_ps(t1, t3, 0x44);
            __m256 z3 = _mm256_shuffle_ps(t1, t3, 0xEE);
            __m256 z4 = _mm256_shuffle_ps(t4, t6, 0x44);
            __m256 z5 = _mm256_shuffle_ps(t4, t6, 0xEE);
            __m256 z6 = _mm256_shuffle_ps(t5, t7, 0x44);
            __m256 z7 = _mm256_shuffle_ps(t5, t7, 0xEE);

            // Scrivi i blocchi trasposti nella matrice di destinazione
            _mm256_storeu_ps(&transposed[col + 0][row], _mm256_permute2f128_ps(z0, z4, 0x20));
            _mm256_storeu_ps(&transposed[col + 1][row], _mm256_permute2f128_ps(z1, z5, 0x20));
            _mm256_storeu_ps(&transposed[col + 2][row], _mm256_permute2f128_ps(z2, z6, 0x20));
            _mm256_storeu_ps(&transposed[col + 3][row], _mm256_permute2f128_ps(z3, z7, 0x20));
            _mm256_storeu_ps(&transposed[col + 4][row], _mm256_permute2f128_ps(z0, z4, 0x31));
            _mm256_storeu_ps(&transposed[col + 5][row], _mm256_permute2f128_ps(z1, z5, 0x31));
            _mm256_storeu_ps(&transposed[col + 6][row], _mm256_permute2f128_ps(z2, z6, 0x31));
            _mm256_storeu_ps(&transposed[col + 7][row], _mm256_permute2f128_ps(z3, z7, 0x31));
        }
    }
    auto end_transpose =  chrono::high_resolution_clock::now();
    chrono::duration<double> duration_transpose=end_transpose - start_transpose;
    return duration_transpose.count();
}

double matTransposeOmp(Matrix &mat, Matrix  transposed, int num_threads) {
    int n = mat.size();
    int block_size;
     chrono::duration<double> duration_transpose;
    if (n <= 64) {
        // La matrice entra interamente nella cache L1
        omp_set_num_threads(num_threads); 
        auto start_transpose = chrono::high_resolution_clock::now();    
        #pragma unroll(8)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {

                transposed[j][i] = mat[i][j];
            }
        }
       auto end_transpose =  chrono::high_resolution_clock::now();
       duration_transpose=end_transpose-start_transpose;
    } 
    else if (n > 64 && n <= 256) {
        block_size = 64;
    } 
    else if (n > 256 && n <= 512) {
        block_size = 128;
    } 
    else if (n > 512 && n <= 1024) {
        block_size = 256;
    } 
    else if (n > 1024 && n <= 2048) {
        block_size = 512;
    } 
    else { // n > 2048
        block_size = 1024;
    }
    
    if (n>64){
        auto start_transpose = chrono::high_resolution_clock::now();
        for (int ii = 0; ii < n; ii += block_size) {
            for (int jj = 0; jj < n; jj += block_size) {
                omp_set_num_threads(num_threads);
                #pragma omp parallel for collapse(2) schedule(guided) 
                for (int i = ii; i < ii + block_size; i++) {
                    for (int j = jj; j < jj + block_size; j ++) { 
                         transposed[j][i] = mat[i][j];
                    }
                }
            }
        }
        auto end_transpose =  chrono::high_resolution_clock::now();
       duration_transpose=end_transpose-start_transpose;
       
    }
    if (duration_transpose.count()==0.0){
        cout<<"ERRORE";
    }
    return duration_transpose.count();
}


timeResult test(int choice, Matrix & matTotran, Matrix & transposed, int num_threads){
    float sequential_time_tran=0.0;
    float sequential_time_check=0.0;
    float imp_time_tran=0.0;
    float imp_time_check=0.0;
    float omp_time_tran=0.0;
    float omp_time_check=0.0;
    switch (choice){
        case 0://sequential
                for(int timer=0; timer<10; timer++){
                     sequential_time_tran+=matTranspose(matTotran, transposed);
                     sequential_time_check+=checkSym(matTotran);
                }
                return timeResult{(sequential_time_tran/10),(sequential_time_check/10)};   
                break;
         case 1://imp
                for (int timer=0.0; timer<10; timer++){
                    imp_time_tran+= matTransposeImp(matTotran, transposed);
                    imp_time_check+=checkSymImp(matTotran); 
                }
                return timeResult{(imp_time_tran/10),(imp_time_check/10)}; 
                break;

         case 2:  //omp   
                for(int timer=0; timer<10; timer++){
                        omp_time_tran+=matTransposeOmp(matTotran, transposed, num_threads);
                        omp_time_check+=checkSymOmp(matTotran,num_threads); 
                    }
                return timeResult{omp_time_tran/10,omp_time_check/10};  
                break;               
    }
    return timeResult{0.0,0.0};
} 


int main(int argv, char *argc[]){ 
     int dim_array[10] = {8,16,32,64,128,256,512,1024,2048,4096};
    //generazione seed randomica
    srand(time(0));
    //definizione mat 1 e 2 
    Matrix matToTran(16, std::vector<float>(16, 0.0f));;
    Matrix transposed(16, std::vector<float>(16, 0.0f));;
    Matrix boh(16, std::vector<float>(16, 0.0f));;
    Matrix boh2(16, std::vector<float>(16, 0.0f));;
    init_matrix(0, matToTran,true);
    init_matrix(0, transposed,true);
    float average_seq_tran[10]={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
    float average_seq_check[10]={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
    float average_omp_check[10][9]= { {0.0f} };
    float average_omp_tran[10][9]={ {0.0f} };
    float average_imp_tran[10]={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};    
    float average_imp_check[10]={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}; 
    timeResult res={0.0,0.0};

    // for(int counter=0;counter<9;counter++){
    //     init_matrix(counter, matToTran,true);
    //     init_matrix(counter, transposed,true);
    //     init_matrix(counter, boh,true);
    //     init_matrix(counter, boh2,true);
    //     matTranspose(matToTran,transposed);
    //     matTransposeImp(matToTran,boh);
    //     matTransposeOmp(matToTran,boh2,4);
        
        
    //     for (int i=0; i<dim_array[counter]; i++)
    //     {
    //         for (int e=0; e<dim_array[counter]; e++)
    //         {
                
    //            if (transposed[i][e]!=boh[i][e] ){
    //             cout<<"dim:" <<dim_array[counter];
    //                cout<<"errore nella trasposizione IMP"<<endl;
    //            }
    //         }
    //     }   


    //     for (int i=0; i<dim_array[counter]; i++)
    //     {
    //         for (int e=0; e<dim_array[counter]; e++)
    //         {
    //            if (transposed[i][e]!=boh2[i][e]){
    //                cout<<"dim:" <<dim_array[counter];
    //                cout<<"errore nella trasposizione OMP"<<" in posizione row: "<<i<<" column: "<<e;;
    //            }
    //         }
    //     }  

    // }
    //  cout<<"test superato"<<endl;
    cout<<"Elaborating matrix: "; 
    for (int counter=1; counter<10; counter++)
    {
        for (int i=0; i<15; i++)
        {
            if (i==0)
            {
                init_matrix(counter, matToTran,true);
                init_matrix(counter, transposed,true);          
            }
            else
            {
                init_matrix(counter, matToTran,false);
                init_matrix(counter, transposed,false);
            }

            res=test(0,matToTran,transposed,0);
            average_seq_tran[counter]+= res.time_tran ;
            average_seq_check[counter]+=res.time_check;
            init_matrix(counter, transposed,false);


            res=test(1,matToTran,transposed,0);
            average_imp_tran[counter]+=res.time_tran ;   
            average_imp_check[counter]+=res.time_check;

            init_matrix(counter, transposed,false);
            for (int num_threads=1; num_threads<9; num_threads++)
            {
                res=test(2,matToTran,transposed,num_threads);
                average_omp_tran[counter][num_threads]+= res.time_tran;
                average_omp_check[counter][num_threads]+= res.time_check;
                init_matrix(counter, transposed,false);
            }
        }
        cout<<dim_array[counter]<<"x"<<dim_array[counter]<<" "; 
    } 
    for (int counter=1; counter<10; counter++)                        
        {
            cout<<"\n                                            test on matrix dimension: "<<dim_array[counter]<<"x"<<dim_array[counter]<<endl;
            cout<<"\n Matrix Transposition                                                           Matrix check"<<endl;
            cout<<"Average time for sequential: "<<average_seq_tran[counter]/15<< "                                                  Average time for sequential:"<<average_seq_check[counter]/15 ;
            cout<<"\nAverage time for iMP: "<<average_imp_tran[counter]/15<<"                                                         Average time for iMP: "<<average_imp_check[counter]/15;
            cout<<"\nPercentuale di miglioramento seq/IMP:"<<((average_seq_tran[counter]-average_imp_tran[counter])/average_seq_tran[counter])*100<<"% "<<"                                      ";
            cout<<"Percentuale di miglioramento seq/IMP:"<<((average_seq_check[counter]-average_imp_check[counter])/average_seq_check[counter])*100<<"% \n";
            for (int num_threads=1; num_threads<9; num_threads++)
            {
                cout<<"Average time for omp with "<<num_threads<<" threads: "<<average_omp_tran[counter][num_threads]/15 <<"                                          "<<"Average time for omp with "<<num_threads<<" threads: "<<average_omp_check[counter][num_threads]/15;
                cout<<"\nPercentuale di miglioramento seq/OMP:"<<((average_seq_tran[counter]-average_omp_tran[counter][num_threads])/average_seq_tran[counter])*100<<"%"<<"                                       Percentuale di miglioramento seq/OMP:"
                <<((average_seq_check[counter]-average_omp_check[counter][num_threads])/average_seq_check[counter])*100<<"% \n"<<endl;
            }

        }
    return 0;
}   
