#include <cuda.h>
#include <iostream>
#include "graph.h"
#include "utilities.h"
#include <stdio.h>
#include <stdlib.h>

__host__ __device__ int BLOCKS(int n){
 	return (int)ceil(((double)(n))/1024);
}

__host__ __device__ int THREADS(int n){
	return n > 1024 ? 1024 : n;
}

__host__ void checkForErr(char* str){
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err!=0){
		printf(RED "ERROR: %s\n" RESET, str);
		printf(RED "ERROR=%d, %s, %s\n" RESET, err, cudaGetErrorName(err), 
			cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	else{
		printf(GRN "SUCCESS: %s\n" RESET, str);
	}
}

void copyToDevice(Node **nodesD, Edge **edgesD,
				 Node **nodesH, Edge **edgesH,
				 int nodes_n, int edges_n){
	cudaMalloc(nodesD, sizeof(Node)*nodes_n);
	cudaMalloc(edgesD, sizeof(Edge)*edges_n);
	cudaMemcpy(*nodesD, *nodesH, sizeof(Node)*nodes_n,
				cudaMemcpyHostToDevice);
	cudaMemcpy(*edgesD, *edgesH, sizeof(Edge)*edges_n,
				cudaMemcpyHostToDevice);
}

void copyNodesToHost(Node **nodesD, Node **nodesH, int nodes_n){
	cudaMemcpy(*nodesH, *nodesD, sizeof(Node)*(nodes_n),
				cudaMemcpyDeviceToHost);
}

void copyEdgesToDevice(Edge **edgesD, Edge **edgesH, int edges_n){
	cudaMemcpy(*edgesD, *edgesH, sizeof(Edge)*(edges_n),
				cudaMemcpyHostToDevice);
}

void copyToHost(Node **nodesD, Edge **edgesD,
				 Node **nodesH, Edge **edgesH,
				 int nodes_n, int edges_n){
	cudaMemcpy(*nodesH, *nodesD, sizeof(Node)*nodes_n,
				cudaMemcpyDeviceToHost);
	cudaMemcpy(*edgesH, *edgesD, sizeof(Edge)*edges_n,
				cudaMemcpyDeviceToHost);
}
