#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>



void cudaWraper(dim3, dim3, cv::Mat&);


void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %lu\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %lu\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %lu\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %lu\n",  devProp.totalConstMem);
    printf("Texture alignment:             %lu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ?"Yes" : "No"));
    return;
}



int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr<<"Za malo argumentow"<<std::endl;
        return 1;
    }

     int devCount;
    cudaGetDeviceCount(&devCount);

    int thread_count;
    try {
        thread_count = std::stoi(argv[1]);
    } catch (std::exception) {
        std::cout<<"nie wlasicwe argumenty"<<std::endl;
        return 4;
    }
    for (int i = 0; i < devCount; ++i)    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        // sprawdzenie maksymalnej liczby watkow na block
        if (thread_count > devProp.maxThreadsPerBlock) {
            std::cerr<<"Za duzo watkow na blok dla tej karty"<<std::endl;
            return 1;
        }
    }
    thread_count = sqrt(thread_count);



    cv::VideoCapture capture(argv[2]);
    if( !capture.isOpened() ) {
        std::cerr<<"Nie mozna otworzyc pliku "<<argv[2]<<std::endl;
        return -1;
    }

     // odczytanie rozdzielczości video
    cv::Size sourceSize = cv::Size((int) capture.get(CV_CAP_PROP_FRAME_WIDTH),
                                     (int) capture.get(CV_CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter outputVideo;
    outputVideo.open(argv[3], capture.get(CV_CAP_PROP_FOURCC), capture.get(CV_CAP_PROP_FPS), sourceSize, true);
    if( !outputVideo.isOpened() ) {
        std::cerr<<"Nie mozna otworzyc pliku "<<argv[3]<<std::endl;
        return -1;
    }
    cv::Mat frame;                      // ramka

    dim3 block(thread_count, thread_count);
    dim3 grid(ceilf((float)(sourceSize.width/thread_count)) + 1 , ceilf((float)(sourceSize.height/thread_count)) + 1);
    cudaEvent_t startEvent, stopEvent;
    // int time  = 0;
    float timeCUDA = 0;
    for(;;) {
        capture >> frame;

        // przerwanie pętli gdy nie ma juz ramek
        if (frame.empty())    {
                break;
        }
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);

        cudaEventRecord(startEvent, 0);

        cudaWraper(grid, block, frame);
        cudaEventRecord(stopEvent, 0);

        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start);
        float elapsed;
                cudaEventSynchronize(stopEvent);
                        cudaEventElapsedTime(&elapsed, startEvent, stopEvent);
        timeCUDA += elapsed;
        // time += duration.count();

        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        outputVideo << frame; //zapis do pliku wyjsciowego


    }
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr) {
        printf("kernel launch failed with error \"%s\".\n",
                                     cudaGetErrorString(cudaerr));
        return -1;
    }
    std::cout<<timeCUDA<<std::endl;

    return 0;
}
