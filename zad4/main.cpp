#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime_api.h>



void cudaWrapper(unsigned  char *);


int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr<<"Za malo argumetow"<<std::endl;
        return -1;
    }
    if (strlen(argv[1]) != 32) {
        std::cerr<<"md5 ma 32 znaki"<<std::endl;
        return -2;
    }

    cudaWrapper(reinterpret_cast<unsigned char*>(argv[1]));


    return 0;
}
