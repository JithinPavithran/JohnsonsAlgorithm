main: graph.o utilities.o algo.o main.o
	nvcc -arch=sm_35 -rdc=true graph.o utilities.o algo.o main.o -o outExe

graph.o: utilities.o graph.cu graph.h
	nvcc -c -arch=sm_35 -rdc=true graph.cu

utilities.o: utilities.cu utilities.h
	nvcc -c -arch=sm_35 -rdc=true utilities.cu

algo.o: utilities.o graph.o algo.cu algo.h
	nvcc -c -arch=sm_35 -rdc=true utilities.o graph.o algo.cu

main.o: main.cu
	nvcc -c -arch=sm_35 -rdc=true main.cu

clean: 
	rm -rf *.o outExe


