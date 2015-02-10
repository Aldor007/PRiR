CC=g++
CFLAGS=-Wall -fopenmp -lpthread -std=c++11
OBJS=main.o
LFLAGS=-fopenmp -lpthread

pi_omp: $(OBJS)
	$(CC) $(LFLAGS) $^ -o $@

main.o: main.cpp
	$(CC) $(CFLAGS) -c $< -o $@

