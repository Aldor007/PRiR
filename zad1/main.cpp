#include <iostream>
#include <cstdlib>
#include <chrono>
#include <string>
#include <omp.h>

void sum(int **A, int **B, int **C, int rozmiar) {
    auto start = std::chrono::system_clock::now();

    // #pragma omp parallel for default(shared)
    int x = 0  ,i = 0,j = 0;
    #pragma omp parallel for default(none) shared(A, B, C) firstprivate(rozmiar)     private(i, j, x)
    for (x = 0; x < 1000; x++)
        for (i=0; i<rozmiar; ++i)
            for (j=0; j<rozmiar; ++j)
                C[i][j]=A[i][j]+B[i][j];
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start);

    std::cout<<duration.count()<<";";
}

void multi(int **A, int **B, int **C, int rozmiar) {
    auto start = std::chrono::system_clock::now();
    // #pragma omp parallel for default(shared)
    int r = 0  ,i = 0,j = 0;
    #pragma omp parallel for default(none) shared(A, B, C) firstprivate(rozmiar)     private(i, j, r)
    for (i=0; i< rozmiar; ++i)
        for (j=0; j < rozmiar;  ++j) {
            for (r = 0; r< rozmiar; ++r)
                C[i][j] = C[i][j] + A[i][r] * B[r][j];
        }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start);
    std::cout<<duration.count()<<"\n";
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout<<"Za malo argumentow"<<std::endl;
        return 1;
    }
    int rozmiar = 0, thread_count = 0;
    try {
        rozmiar = std::stoi(argv[2]);
        thread_count = std::stoi(argv[1]);
    } catch (std::exception) {
        std::cout<<"nie wlasicwe argumenty"<<std::endl;
        return 4;
    }
    omp_set_num_threads(thread_count);

    int **A, **B, **C;
    try {
        A = new int*[rozmiar];
        B = new int*[rozmiar];
        C = new int*[rozmiar];
    } catch( std::bad_alloc) {
        std::cout<<"Za duza tablica";
        return 2;
    }

    try {
        for (int i = 0; i< rozmiar; ++i) {
            A[i] = new int[rozmiar];
            B[i] = new int[rozmiar];
            C[i] = new int[rozmiar];
        }
    } catch( std::bad_alloc) {
        std::cout<<"Za duza tablica"<<std::endl;
        return 2;
    }

    for (int i = 0; i < rozmiar; ++i)
        for (int j = 0; j < rozmiar; ++j) {
            A[i][j] = rand() % 1000;
            B[i][j] = rand() % 1000;
            C[i][j] = 0;
        }
    sum(A, B, C, rozmiar);
    for (int i = 0; i < rozmiar; ++i)
        for (int j = 0; j < rozmiar; ++j) {
            C[i][j] = 0;
        }
    multi(A, B, C, rozmiar);

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
