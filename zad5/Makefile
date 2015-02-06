CC=mpic++
CFLAGS=-Wall -std=c++11
OBJS=main.o
LFLAGS=

miller_rabin: $(OBJS)
	$(CC) $(LFLAGS) $^ -o $@

main.o: main.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean: 
	rm main.o miller_rabin

