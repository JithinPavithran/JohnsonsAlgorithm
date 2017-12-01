#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <climits>
#include "graph.h"
#include "utilities.h"

using namespace std;

/* The function allocates memory required for the graph in CPU.
   It allocates extra memory for BellmanFord Computations as well.
*/
void createGraph(Node **nodes, Edge **edges, int nodes_n, int edges_n){
	*nodes = (Node*) malloc(sizeof(Node)*(nodes_n+1));
	if(*nodes==NULL){
		printf(RED "Failed to allocate memory for Nodes" RESET);
		exit(0);
	}
	*edges = (Edge*) malloc(sizeof(Edge)*(edges_n+nodes_n));
	if(*edges==NULL){
		printf(RED "Failed to allocate memory for Edges\n" RESET);
		exit(0);
	}
}

void loadGraph(Node **nodes, Edge **edges, int nodes_n, int edges_n){
	int v1, v2, e, e_count=0;
	cin>>v1>>v2>>e;
	for(int i=0; i<nodes_n && e_count<edges_n; ++i){
		(*nodes)[i].id     = i;
		(*nodes)[i].offset = e_count;
		(*nodes)[i].dist   = INT_MAX;
		(*nodes)[i].dirty  = 0;
		(*nodes)[i].preNode= i;
		(*nodes)[i].locked = 0; // NOT locked
		while(v1==i && e_count<edges_n){
			(*edges)[e_count].wt   = e;
			(*edges)[e_count].dest = v2;
			(*edges)[e_count].src  = v1;
			++e_count;
			cin>>v1>>v2>>e;
		}
	}
	for(int i=0; i<nodes_n; ++i){
		if((*nodes)[i].id != i){
			(*nodes)[i].id     = i;
			(*nodes)[i].offset = e_count;
			(*nodes)[i].dist   = INT_MAX;
			(*nodes)[i].dirty  = 0;
			(*nodes)[i].preNode= i;
			(*nodes)[i].locked = 0; // NOT locked			
		}
	}
	checkForErr("Graph Loaded");
}

__host__ __device__ void printEdges(Edge *edges, int edges_n){
	printf("Graph:\n");
	printf("src\tdest\tedge.wt\tdist(dest)\n");
	int i;
	for(i=0; i<edges_n; ++i)
		printf("%d\t%d\t%d\n", edges[i].src, edges[i].dest, edges[i].wt);
}

__host__ __device__ void printNodes(Node *nodes, int nodes_n, int final=1){
	printf("Nodes:\n");
	printf("id\tdist\tpreNode\n");
	int i;
	for(i=0; i<nodes_n; ++i){
		if(final || nodes[i].dist<INT_MAX)
			printf("%d\t%d\t%d\n", nodes[i].id, nodes[i].dist, nodes[i].preNode);
		else
			printf("%d\t%s\n", nodes[i].id, "Can't reach");
	}
}

__global__ void printNodesFromGPU(Node *nodes, int nodes_n){
	printf("%s\n", "Printing from GPU:");
	if(threadIdx.x==0 && blockIdx.x==0){
		cudaDeviceSynchronize();
		printNodes(nodes, nodes_n);
	}
}

__global__ void printEdgesFromGPU(Edge *edges, int edges_n){
	printf("%s\n", "Printing from GPU:");
	if(threadIdx.x==0 && blockIdx.x==0)
		cudaDeviceSynchronize();
		printEdges(edges, edges_n);
}

__global__ void setDist(Node *nodes, int n, int val){
	if(threadIdx.x==0){
		nodes[n].dist    = val;
		nodes[n].preNode = n;
	}
}

__host__ void printResult(Node *nodes, int nodes_n, int SRC_NODE){
	FILE *fp;
	fp = fopen("result.txt", "w");
	fprintf(fp, "Distance to nodes from node:%d\n", SRC_NODE);
	fprintf(fp, "id\tdist\tpreNode\n");
	int i;
	for(i=0; i<nodes_n; ++i){
		if(nodes[i].dist<INT_MAX)
			fprintf(fp, "%d\t%d\t%d\n",
					 nodes[i].id, nodes[i].dist, nodes[i].preNode);
		else
			fprintf(fp, "%d\t%s\n", nodes[i].id, "Can't reach");
	}
	fclose(fp);
}