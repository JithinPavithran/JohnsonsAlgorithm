#include <iostream>
#include "graph.h"
#include "algo.h"
#include <stdio.h>
#include <stdlib.h>
#include <climits>
#include "utilities.h"

using namespace std;


__global__ void prepareForBF(Node* nodes, Edge* edges,
						 int nodes_n, int edges_n){
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id<nodes_n){
		edges[edges_n+id].dest = id;
		edges[edges_n+id].wt   = 0;
		edges[edges_n+id].src  = nodes_n;
	}
	if(id==0){	
		nodes[nodes_n].id      = nodes_n;
		nodes[nodes_n].offset  = edges_n;
		nodes[nodes_n].dist    = 0;
		nodes[nodes_n].dirty   = 0;
	}
}

__global__ void setDist(Node* nodes, Edge* edges,
						 int nodes_n, int edges_n, int val){
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id<nodes_n){
		nodes[id].dist    = val;
		nodes[id].preNode = id;
	}
}

__global__ void setMin(Node* nodes, Edge* edges,
						 int nodes_n, int edges_n){
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id<edges_n){
		if(nodes[edges[id].dest].dist 
			> nodes[edges[id].src].dist+edges[id].wt){
			nodes[edges[id].dest].dist = nodes[edges[id].src].dist+edges[id].wt;
		}
	}
}

__global__ void negetiveCycleCheck(Node* nodes, Edge* edges,
						 int nodes_n, int edges_n, int *negetiveCycles){
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id<edges_n){
		if(nodes[edges[id].dest].dist 
			> nodes[edges[id].src].dist+edges[id].wt){
			*negetiveCycles = 1;
		}
	}
}


void BellmanFord(Node* nodes, Edge* edges, int nodes_n, int edges_n){
	setDist<<<BLOCKS(nodes_n), THREADS(nodes_n)>>>(
				nodes, edges, nodes_n, edges_n, 0);
	cudaDeviceSynchronize();
	checkForErr("Set dist to Max (0)");

	for(int i=0; i<nodes_n; ++i){
		setMin<<<BLOCKS(edges_n), THREADS(edges_n)>>>(
					nodes, edges, nodes_n, edges_n);
	}
	cudaDeviceSynchronize();
	checkForErr("Set dist to Min");

	int *negetiveCycles;
	cudaHostAlloc(&negetiveCycles, sizeof(int), 0);
	negetiveCycleCheck<<<BLOCKS(edges_n), BLOCKS(edges_n)>>>(
					nodes, edges, nodes_n, edges_n, negetiveCycles);
	cudaDeviceSynchronize();
	if(*negetiveCycles!=0){
		printf(RED "\n\nERROR: Negetive Cycles detected\n\n" RESET);
		exit(EXIT_SUCCESS);
	}
}


__global__ void reweight(Node *nodes, Edge *edges, int nodes_n, int edges_n){
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id<edges_n){
		edges[id].wt = edges[id].wt + nodes[edges[id].src].dist
									- nodes[edges[id].dest].dist;
	}
}

/*
nodes[preNode]                                             --------------
             |                                     /------>| futfutNode |
             |                          ----------/        --------------
             V                    /---->| futNode |
    -----------       -----------/      -----------
--->| preNode |------>| currNode |----->| futNode |
    -----------   ^   -----------\      -----------
                  |               ----->| futNode |
                  |                     -----------
             preEdge             \      /
                                currEdges
*/
/* Find minimum distance to all nodes directly reachable from preNode_id
   Eg: if currNode=preNode_id, find minimum distance to futNode(s)
*/
__global__ void find_min_dist(Node* nodes, Edge* edges, int nodes_n, int edges_n, int preNode_id){
	int id = threadIdx.x + blockIdx.x * blockDim.x;
/*spawnEdges: number of edges originaing from preNode*/
	int preEdges_n  = preNode_id < nodes_n-1 ?
					  nodes[preNode_id+1].offset - nodes[preNode_id].offset :
					  edges_n - nodes[preNode_id].offset;
	// printf("preNode:%d, preEdges_n:%d\n", preNode_id, preEdges_n);
	if(id < preEdges_n){
		Edge *preEdge  = &(edges[nodes[preNode_id].offset + id]);
		Node *currNode = &(nodes[(*preEdge).dest]);
		Node *preNode  = &(nodes[preNode_id]);
		// printf("%d --> %d\n", (*preNode).id, (*currNode).id);

		do{}while(atomicCAS(&((*currNode).locked), 0, 1)==1);

		int old_dist   = (*currNode).dist;
		int new_dist   = (*preNode).dist + (*preEdge).wt;
		int currEdges_n;
		if(old_dist > new_dist){
			(*currNode).dist    = new_dist;
			(*currNode).dirty   = 0;
			(*currNode).preNode = preNode_id;
			(*currNode).locked  = 0;

			// Setting all reachable nodes to dirty
			currEdges_n = (*currNode).id < nodes_n-1 ?
						  (*(currNode+1)).offset - (*currNode).offset :
						  edges_n - (*currNode).offset;
			for(int i=(*currNode).offset; i<currEdges_n; ++i){
				nodes[edges[i].dest].dirty = 1;
			}
			// launch kernels on them
			// printf("Node:%d, currEdges:%d\n", (*currNode).id, currEdges_n);
			find_min_dist<<<BLOCKS(currEdges_n), THREADS(currEdges_n)>>>(
							nodes, edges, nodes_n, edges_n, (*currNode).id);
			cudaDeviceSynchronize();
		}
		else{
			(*currNode).locked  = 0;
		}
	}
}

 void findDist(Node *nodes, Edge *edges,
					 int nodes_n, int edges_n, int SRC){
	for(int id=0; id<nodes_n; ++id){
		int currNode   = id;
		nodes[id].dist = 0;
		while(nodes[currNode].id != SRC) {
			if(nodes[currNode].preNode==nodes[currNode].id){
				nodes[id].dist = INT_MAX;
				break;
			}
			int preNode = nodes[currNode].preNode;
			int edge_spawn_n  = preNode < nodes_n-1 ?
								nodes[preNode+1].offset - nodes[preNode].offset :
								edges_n - nodes[preNode].offset;
			int j = nodes[preNode].offset;
			for(;j<nodes[preNode].offset+edge_spawn_n; ++j){
				if(edges[j].dest==currNode){
					nodes[id].dist+=edges[j].wt;
					break;
				}
			}
			if(j==nodes[preNode].offset+edge_spawn_n){
				printf(RED "Error Finding path distance\n" RESET);
				return;
			}
			currNode = preNode;
		}
	}
}

 __global__ void findDistD(Node *nodes, Edge *edges,
					 int nodes_n, int edges_n, int SRC){
 	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id<nodes_n){
		int currNode   = id;
		nodes[id].dist = 0;
		while(nodes[currNode].id != SRC) {
			if(nodes[currNode].preNode==nodes[currNode].id){
				nodes[id].dist = INT_MAX;
				break;
			}
			int preNode = nodes[currNode].preNode;
			int edge_spawn_n  = preNode < nodes_n-1 ?
								nodes[preNode+1].offset - nodes[preNode].offset :
								edges_n - nodes[preNode].offset;
			int j = nodes[preNode].offset;
			for(;j<nodes[preNode].offset+edge_spawn_n; ++j){
				if(edges[j].dest==currNode){
					nodes[id].dist+=edges[j].wt;
					break;
				}
			}
			if(j==nodes[preNode].offset+edge_spawn_n){
				printf(RED "Error Finding path distance\n" RESET);
				return;
			}
			currNode = preNode;
		}
	}
}


