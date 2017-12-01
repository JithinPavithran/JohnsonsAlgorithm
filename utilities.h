#ifndef UTILITIES_H 
#define UTILITIES_H


void copyToDevice(Node **nodesD, Edge **edgesD,
				 Node **nodesH, Edge **edgesH,
				 int nodes_n, int edges_n);

__host__ void checkForErr(char* str);

void copyNodesToHost(Node **nodesD, Node **nodesH, int nodes_n);
void copyEdgesToDevice(Edge **edgesD, Edge **edgesH, int edges_n);

void copyToHost(Node **nodesD, Edge **edgesD,
				 Node **nodesH, Edge **edgesH,
				 int nodes_n, int edges_n);

#define RED   "\x1B[31m"
#define GRN   "\x1B[32m"
#define YEL   "\x1B[33m"
#define RESET "\x1B[0m"

__host__ __device__ int THREADS(int n);
__host__ __device__ int BLOCKS(int n);


#endif
