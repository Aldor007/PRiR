#ifndef CUDA_siniERNEL_MD5
#define CUDA_siniERNEL_MD5
#define MD5_PER_KERNEL 400
#define CHARS_LEN 63

#include <cstdio>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>

/* tablica znakow */
static unsigned char chars_table_cpu[] = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a','b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
    'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
};


/* tablica na wartosci intowe celu */
__constant__ uint target[4];
/* tablica dla CUDY ze znakami */
__constant__ unsigned  char chars_table[CHARS_LEN];

/* funkcje md5 */
__device__ __host__
inline void FF (uint &a, uint &b, uint &c, uint &d, const uint x, const uint s, const uint ac) {
    a += (((b) & (c)) | ((~b) & (d))) + x + ac;
/* przesuniecie o  s bitow w lewo, to bylo z lewej wraca z prawej*/
    a = (a << s) | (a >> (32 - s));
    a += b;
}


__device__ __host__
inline void GG (uint &a, uint &b, uint &c, uint &d, const uint x, const uint s, const uint ac) {
    a += (((b) & (d)) | ((c) & (~d))) + x + ac;
    a = (a << s) | (a >> (32 - s));
    a += b;
}

__device__ __host__
inline void HH (uint &a, uint &b, uint &c, uint &d, const uint x, const uint s, const uint ac) {
    a += ((b) ^ (c) ^ (d)) + x + ac;
    a = (a << s) | (a >> (32 - s));
    a += b;
}


__device__ __host__
inline void II (uint &a, uint &b, uint &c, uint &d, const uint x, const uint s, const uint ac) {
    a += ((c) ^ ((b) | (~d))) + x + ac;
    a = (a << s) | (a >> (32 - s));
    a += b;
}

__device__ __host__
inline void md5 (uint * in, uint &a, uint &b, uint &c, uint &d) {
    /* magiczne stale z algorytmu*/
    const uint a0 = 0x67452301;
    const uint b0 = 0xEFCDAB89;
    const uint c0 = 0x98BADCFE;
    const uint d0 = 0x10325476;
    
    a = a0;
    b = b0;
    c = c0;
    d = d0;
    //  magiczne dane z alg, block 512 bitowy, przesuniecie, 32 bity z czesci ulamkowej  |sini|
    FF(a, b, c, d, in[0],  7, 0xd76aa478);
    FF(d, a, b, c, in[1],  12, 0xe8c7b756);
    FF(c, d, a, b, in[2],  17, 0x242070db);
    FF(b, c, d, a, in[3],  22, 0xc1bdceee);
    FF(a, b, c, d, in[4],  7, 0xf57c0faf);
    FF(d, a, b, c, in[5],  12, 0x4787c62a);
    FF(c, d, a, b, in[6],  17, 0xa8304613);
    FF(b, c, d, a, in[7],  22, 0xfd469501);
    FF(a, b, c, d, in[8],  7, 0x698098d8);
    FF(d, a, b, c, in[9],  12, 0x8b44f7af);
    FF(c, d, a, b, in[10],  17, 0xffff5bb1);
    FF(b, c, d, a, in[11],  22, 0x895cd7be);
    FF(a, b, c, d, in[12],  7, 0x6b901122);
    FF(d, a, b, c, in[13],  12, 0xfd987193);
    FF(c, d, a, b, in[14],  17, 0xa679438e);
    FF(b, c, d, a, in[15],  22, 0x49b40821);

    GG(a, b, c, d, in[1],  5, 0xf61e2562);
    GG(d, a, b, c, in[6],  9, 0xc040b340);
    GG(c, d, a, b, in[11],  14, 0x265e5a51);
    GG(b, c, d, a, in[0],  20, 0xe9b6c7aa);
    GG(a, b, c, d, in[5],  5, 0xd62f105d);
    GG(d, a, b, c, in[10],  9, 0x2441453);
    GG(c, d, a, b, in[15],  14, 0xd8a1e681);
    GG(b, c, d, a, in[4],  20, 0xe7d3fbc8);
    GG(a, b, c, d, in[9],  5, 0x21e1cde6);
    GG(d, a, b, c, in[14],  9, 0xc33707d6);
    GG(c, d, a, b, in[3],  14, 0xf4d50d87);
    GG(b, c, d, a, in[8],  20, 0x455a14ed);
    GG(a, b, c, d, in[13],  5, 0xa9e3e905);
    GG(d, a, b, c, in[2],  9, 0xfcefa3f8);
    GG(c, d, a, b, in[7],  14, 0x676f02d9);
    GG(b, c, d, a, in[12],  20, 0x8d2a4c8a);

    HH(a, b, c, d, in[5],  4, 0xfffa3942);
    HH(d, a, b, c, in[8],  11, 0x8771f681);
    HH(c, d, a, b, in[11],  16, 0x6d9d6122);
    HH(b, c, d, a, in[14],  23, 0xfde5380c);
    HH(a, b, c, d, in[1],  4, 0xa4beea44);
    HH(d, a, b, c, in[4],  11, 0x4bdecfa9);
    HH(c, d, a, b, in[7],  16, 0xf6bb4b60);
    HH(b, c, d, a, in[10],  23, 0xbebfbc70);
    HH(a, b, c, d, in[13],  4, 0x289b7ec6);
    HH(d, a, b, c, in[0],  11, 0xeaa127fa);
    HH(c, d, a, b, in[3],  16, 0xd4ef3085);
    HH(b, c, d, a, in[6],  23, 0x4881d05);
    HH(a, b, c, d, in[9],  4, 0xd9d4d039);
    HH(d, a, b, c, in[12],  11, 0xe6db99e5);
    HH(c, d, a, b, in[15],  16, 0x1fa27cf8);
    HH(b, c, d, a, in[2],  23, 0xc4ac5665);

    II(a, b, c, d, in[0],  6, 0xf4292244);
    II(d, a, b, c, in[7],  10, 0x432aff97);
    II(c, d, a, b, in[14],  15, 0xab9423a7);
    II(b, c, d, a, in[5],  21, 0xfc93a039);
    II(a, b, c, d, in[12],  6, 0x655b59c3);
    II(d, a, b, c, in[3],  10, 0x8f0ccc92);
    II(c, d, a, b, in[10],  15, 0xffeff47d);
    II(b, c, d, a, in[1],  21, 0x85845dd1);
    II(a, b, c, d, in[8],  6, 0x6fa87e4f);
    II(d, a, b, c, in[15],  10, 0xfe2ce6e0);
    II(c, d, a, b, in[6],  15, 0xa3014314);
    II(b, c, d, a, in[13],  21, 0x4e0811a1);
    II(a, b, c, d, in[4],  6, 0xf7537e82);
    II(d, a, b, c, in[11],  10, 0xbd3af235);
    II(c, d, a, b, in[2],  15, 0x2ad7d2bb);
    II(b, c, d, a, in[9],  21, 0xeb86d391);

    a += a0;
    b += b0;
    c += c0;
    d += d0;
}


__device__ __host__ bool incrementIndex(unsigned int * brute, int setLen, int wordwordLen, int incrementBy)
{
    int i = 0;
    while(incrementBy > 0 && i < wordwordLen) {
            int add = incrementBy + brute[i];
            brute[i] = add % setLen;
            incrementBy = add / setLen;
            i++;
     }
    return incrementBy != 0; // koniec zakresu
}

void gpu_init (unsigned int * abcd) {
    cudaMemcpyToSymbol(target, abcd, sizeof(target));
    cudaMemcpyToSymbol(chars_table, chars_table_cpu, sizeof(chars_table_cpu));
}

__global__
void md5_bruteforce (uint * cudaIndex, int wordLen, int * res, char * result) {

    uint thread_id = (blockIdx.x*blockDim.x + threadIdx.x);

    uint localBrute[16];// 16*32 = 512

    for(int i = 0; i < wordLen; i++) {
        localBrute[i] = cudaIndex[i];
    }

    *res = -1;
    incrementIndex(localBrute, CHARS_LEN, wordLen, thread_id);
    int i = 0;
    for(int j = 0; j < MD5_PER_KERNEL; j++)  {
        uint a, b, c, d, in[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (i = 0; i <wordLen; i++) {
            in[i / 4] |= chars_table[localBrute[i]]<< ((i % 4) * 8);
        }
        in[i / 4] |= 0x80 << ((i % 4) * 8);
        in[14] = wordLen * 8; // 448 bit - dlugosc slowa

        md5(in, a, b, c, d);
        if (target[0] == a && target[1] == b && target[2] == c && target[3] == d) {
            (* res) = thread_id; // flaga
            for (int i =0; i < wordLen; i++) {
                result[i] = chars_table[localBrute[i]]; // zapisanie wyniku
            }

            return;


        }

        incrementIndex(localBrute, CHARS_LEN, wordLen, blockDim.x * gridDim.x);
    }

}

uint unhex(unsigned char x) {
    if(x <= 'F' && x >= 'A')  {
        return  (uint)(x - 'A' + 10);
    } else if(x <= 'f' && x >= 'a')  {
        return (uint)(x - 'a' + 10);
    }
    else if(x <= '9' && x >= '0') {
        return (uint)(x - '0');
    }
    return 0;
}

void md5_to_ints(unsigned char* md5, uint *abcd) {
    uint v0 = 0, v1 = 0, v2 = 0, v3 = 0;
    int i = 0;
    for(i = 0; i < 32; i+=2) {
        uint first = unhex(md5[i]);
        uint second = unhex(md5[i+1]);
        uint both = first * 16 + second;
        both = both << 24;
        if(i < 8) {
            v0 = (v0 >> 8 ) | both;
        } else if (i < 16) {
            v1 = (v1 >> 8) | both;
        } else if (i < 24) {
            v2 = (v2 >> 8) | both;
        } else if(i < 32) {
            v3 = (v3 >> 8) | both;
        }
    }

    abcd[0] = v0;
    abcd[1] = v1;
    abcd[2] = v2;
    abcd[3] = v3;
}





void cudaWrapper(unsigned char * md5Data) {
    uint * abcd = new uint[4];
    md5_to_ints(md5Data, abcd);
    /* po to zeby bylo szybicje, nie trzeba zameiniac na hexa */
    gpu_init(abcd);
    bool finished = false;
    uint wordwordLen = 1;

    char * result;
    int ret = -1;
    int *ret_ptr;
    uint currentIndex[16] ={0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    uint *  cudaIndex;
    cudaMalloc(&cudaIndex, sizeof(currentIndex));
    cudaMalloc(&ret_ptr, sizeof(ret));
    cudaMalloc(&result, 8 * sizeof(char));
    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    
    do {
        cudaMemcpy(cudaIndex, &currentIndex, sizeof(currentIndex), cudaMemcpyHostToDevice);
        cudaMemcpy(ret_ptr, &ret, sizeof(ret), cudaMemcpyHostToDevice);

        dim3 dimGrid(4);
        dim3 dimBlock(1024);

        md5_bruteforce<<<dimGrid, dimBlock>>>(cudaIndex, wordwordLen, ret_ptr, result);
        cudaMemcpy(&ret, ret_ptr, sizeof(ret), cudaMemcpyDeviceToHost);
        cudaMemcpy(currentIndex, cudaIndex, sizeof(currentIndex), cudaMemcpyDeviceToHost);
        if (ret != -1) {
            char result_cpu[8];
            cudaMemcpy(result_cpu, result, 8 * sizeof(char), cudaMemcpyDeviceToHost);
            for (int i = 0; i< wordwordLen; i++)
                std::cout<<result_cpu[i];
            std::cout<<std::endl;


            finished = true;
            break;
        }

        finished = incrementIndex(currentIndex, CHARS_LEN, wordwordLen,MD5_PER_KERNEL);

        cudaError_t err = cudaGetLastError();
        if(cudaSuccess != err)  {
            std::cerr<<"Cuda error "<<cudaGetErrorString(err);
            exit(-1);
        }

        /* skonczony dany przedial idziemy dalej */
        if (finished) {
            wordwordLen++;
            finished = false;
        }
    } while(wordwordLen <= 8);

    cudaThreadSynchronize();
    cudaFree(&cudaIndex);
    cudaFree(&ret_ptr);
    cudaFree(&result);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    std::cout<<time<<std::endl;
}



#endif
