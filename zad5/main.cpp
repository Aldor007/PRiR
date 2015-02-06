#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <stdexcept>
#include <chrono>

#include <mpi.h>

#define MY_TAG 12
#define RESPONSE 1
#define ll long long


std::vector<long long> loadNumbers(char * fileName) {
    std::vector<long long> result;
    std::ifstream in(fileName);
    if (!in.good()) {
        in.close();
        throw std::invalid_argument("Podany plik nie istnieje!");
    }
    long long tmp;
    while(in>>tmp) {
        result.push_back(tmp);
    }
    in.close();
    return std::move(result);

}

ll mulmod(ll a, ll b, ll mod) {
    ll x = 0,y = a % mod;
    while (b > 0)
    {
        if (b % 2 == 1)
        {
            x = (x + y) % mod;
        }
        y = (y * 2) % mod;
        b /= 2;
    }
    return x % mod;
}
ll modulo(ll base, ll exponent, ll mod) {
    ll x = 1;
    ll y = base;
    while (exponent > 0)
    {
        if (exponent % 2 == 1)
            x = (x * y) % mod;
        y = (y * y) % mod;
        exponent = exponent / 2;
    }
    return x % mod;
}

int testMillerRabin(ll p,int iteration)
{
    srand(time(NULL));
    if (p < 2) {
        return 0;
    }
    if (p != 2 && p % 2 == 0)  {
        return 0;
    }
    ll s = p - 1;
    while (s % 2 == 0) {
        s /= 2;
    }
    for (int i = 0; i < iteration; i++) {

        ll a = rand() % (p - 1) + 1, temp = s;
        ll mod = modulo(a, temp, p);
        while (temp != p - 1 && mod != 1 && mod != p - 1)    {
            mod = mulmod(mod, mod, p);
            temp *= 2;
        }
        if (mod != p - 1 && temp % 2 == 0)   {
            return 0;
        }
    }
    return 1;
}



int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr<<"Za malo argumentow"<<std::endl;
        return -1;
    }
    int precision = std::atoi(argv[2]);
   // Initialize the MPI environment
    MPI_Init(NULL, NULL);
      // Find out rank, size
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size == 1) {
        std::cerr<<"Za malo procesow!"<<std::endl;
        return -3;

    }
    if (world_rank == 0) {
        std::vector<long long> data;
        try {
            data = loadNumbers(argv[1]);
        } catch(std::invalid_argument &e) {
            std::cerr<<e.what()<<std::endl;
            return -2;
        }

        std::vector<MPI_Request*> requests;
        long long * taskMap = new long long[world_size];
        int rank  = 1;
        auto start = std::chrono::system_clock::now();
        while( data.size() != 0 && rank < world_size) {
            MPI_Request myRequest;
            taskMap[rank] = data.back();
            data.pop_back();
            MPI_Isend(&taskMap[rank], 1,  MPI_LONG, rank, MY_TAG, MPI_COMM_WORLD, &myRequest);
            // MPI_Send(&taskMap[rank], 1,  MPI_LONG, rank, MY_TAG, MPI_COMM_WORLD);
        #ifdef DEBUG
            std::cout<<"data size = "<<data.size()<<"  process = "<<world_size<<std::endl;
        #endif

            requests.push_back(&myRequest);
            rank++;


        }
        // for (auto it = requests.end(); it != requests.begin(); --it) {
        for (int i = 0, len = requests.size(); i < len; ++i){
            MPI_Status myStatus;
            MPI_Wait(requests[i], &myStatus);
        }
        int toRecv = data.size() + rank - 1;
        // while (rank >= 0 || data.size() != 0 ) {
        while (toRecv != 0) {
            MPI_Status responseStatus;
            int result = 0;
            MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE, RESPONSE, MPI_COMM_WORLD, &responseStatus);
            std::cout<<taskMap[responseStatus.MPI_SOURCE]<<": "<<(result?"pierwsza":"zlozona")<<std::endl;
            if (data.size() != 0) {
                taskMap[responseStatus.MPI_SOURCE] = data.back();
                data.pop_back();
                #ifdef DEBUG
                std::cout<<"Sending to "<<responseStatus.MPI_SOURCE<<" "<<taskMap[responseStatus.MPI_SOURCE]<<std::endl;
                #endif

                MPI_Send(&taskMap[responseStatus.MPI_SOURCE], 1, MPI_LONG, responseStatus.MPI_SOURCE, MY_TAG, MPI_COMM_WORLD);

            }
            toRecv--;

        }

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start);
        std::cout<<"Czas: " << duration.count()<<" ms"<<std::endl;
        return 0;


    } else {
        while (1) {
            long long number;
            MPI_Status recvStatus;
            #ifdef DEBUG
            std::cout<<"wating for data "<<world_rank<<"\n";
            #endif
            MPI_Recv(&number, 1, MPI_LONG, MPI_ANY_SOURCE, MY_TAG, MPI_COMM_WORLD, &recvStatus);
            int result = testMillerRabin(number, precision);
            #ifdef DEBUG
            std::cout<<"number -----------"<<number<<"============ result "<<result<< std::endl;
            #endif
            MPI_Send(&result, 1, MPI_INT, 0, RESPONSE, MPI_COMM_WORLD);

        }


    }



    MPI_Finalize();
    return 0;
}
