#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <omp.h>
#include <chrono>
 
int main(int argc, char **argv)
{
     srand(time(NULL));
     int count, ncount;
     int ncycle = 0;
     double x,y;
     float pi;
     int i = 0;
     unsigned seed;
     if (argc != 3)
     {
	std::cout << "Zla liczba argumentow!" << std::endl;
	return 1;
     }

     try 
     {
	count = atoi(argv[1]);
	ncount = atoi(argv[2]);
     }
     catch (std::exception) 
     {
        std::cout<<"nie wlasicwe argumenty"<<std::endl;
        return 2;
     }
     omp_set_num_threads(count);

     auto start = std::chrono::system_clock::now();
     #pragma omp parallel shared(ncount) private(i, x, y, seed)
     { 
       seed = time(0); 
       #pragma omp for reduction(+ : ncycle)
       for(i = 0; i < ncount; i++)
       {           
          x = ((double)rand_r(&seed) / (RAND_MAX))*2 - 1;
          y = ((double)rand_r(&seed) / (RAND_MAX))*2 - 1;
          if(x*x + y*y <= 1)
          {
              ncycle+=1;
          }
       }
     }
     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start);     
     pi = 4. * ncycle / ncount;
     std::cout<<duration.count()<<";";
     std::cout<<pi<<";"<<std::endl;
}
