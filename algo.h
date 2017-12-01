#ifndef BELLMAN_H 
#define BELLMAN_H


__global__ void prepareForBF(Node* nodes, Edge* edges,
							 int nodes_n, int edges_n);

void BellmanFord(Node* nodes, Edge* edges, int nodes_n, int edges_n);

__global__ void reweight(Node *nodes, Edge *edges,
						 int nodes_n, int edges_n);

__global__ void setDist(Node* nodes, Edge* edges,
						 int nodes_n, int edges_n, int val);

__global__ void find_min_dist(Node* nodes, Edge* edges,
						 int nodes_n, int edges_n, int preNode_id);

void findDist(Node *nodes, Edge *edges,
					 int nodes_n, int edges_n, int SRC);

 __global__ void findDistD(Node *nodes, Edge *edges,
					 int nodes_n, int edges_n, int SRC);


#endif
