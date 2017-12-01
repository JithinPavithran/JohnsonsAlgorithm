#ifndef GRAPH_H 
#define GRAPH_H


struct Edge{
	int wt;
	int src;
	int dest;
};

struct Node{
	// offset: offset to vertices starting from this node.
	int id, offset, dist, dirty, preNode, locked;
	// locked if locked=1
};

void createGraph(Node **nodes, Edge **edges, int nodes_n, int edges_n);

void loadGraph(Node **nodes, Edge **edges, int nodes_n, int edges_n);

__host__ __device__ void printEdges(Edge *edges, int edges_n);

__global__ void printEdgesFromGPU(Edge *edges, int edges_n);

__host__ __device__ void printNodes(Node*, int, int);

__global__ void printNodesFromGPU(Node *nodes, int nodes_n);

__global__ void setDist(Node *nodes, int n, int val);

__host__ void printResult(Node *nodes, int nodes_n, int SRC_NODE);

#endif
