/*   
 *	 Cuda based implementation of Johnson's algorithm
 *   
 * 	 Auther: Jithin Pavithran
 *   Mail  : jithinpavithran.public@gmail.com
 *	 
 *	 Input : stdin (use ./executable < inputfile)
 *   Output: (will be printed to) result.txt	
 *	 Input format:
 * 			<No of nodes> <No of edges>
 * 			<starting node> <destination node> <weight of edge>
 * 			<starting node> <destination node> <weight of edge>
 * 			<starting node> <destination node> <weight of edge>
 *		Note: Only integers allowed.
 *				~ +ve for node number (starting from 1)
 *				~ +ve or -ve for edge weight
 * 			  Edges are expected to be sorted according to source edge
 *
 */

#include <stdio.h>
#include <iostream>
#include <climits>
#include "graph.h"
#include "algo.h"
#include "utilities.h"

using namespace std;

int main(){
	int nodes_n, edges_n, SRC_NODE=0;
	Edge *edgesH, *edgesD;
	Node *nodesH, *nodesD;
	cin>>nodes_n>>edges_n>>SRC_NODE;
	printf(YEL "No of Nodes:%d, Edges:%d, Source Node:%d\n" YEL,
			nodes_n, edges_n, SRC_NODE);
	createGraph(&nodesH, &edgesH, nodes_n, edges_n);
	loadGraph(&nodesH, &edgesH, nodes_n, edges_n);

	copyToDevice(&nodesD, &edgesD, &nodesH, &edgesH,
				 nodes_n+1, edges_n+nodes_n);
	checkForErr("Copying data to Device");

	prepareForBF<<<BLOCKS(nodes_n), THREADS(nodes_n)>>>(
				nodesD, edgesD, nodes_n, edges_n);
	cudaDeviceSynchronize();
	checkForErr("Prepare for BellmanFord");

	BellmanFord(nodesD, edgesD, nodes_n+1, edges_n+nodes_n);
	cudaDeviceSynchronize();
	checkForErr("BellmanFord");

	reweight<<<BLOCKS(edges_n), THREADS(edges_n)>>>(
				nodesD, edgesD, nodes_n, edges_n);
	cudaDeviceSynchronize();
	checkForErr("Reweight");

	/*Starting Dijkstra's*/
	setDist<<<BLOCKS(nodes_n), THREADS(nodes_n)>>>(
				nodesD, edgesD, nodes_n, edges_n, INT_MAX);
	cudaDeviceSynchronize();
	checkForErr("setDist");

	int spawnEdges = SRC_NODE < nodes_n-1 ?
					 nodesH[SRC_NODE+1].offset-nodesH[SRC_NODE].offset :
					 edges_n - nodesH[SRC_NODE].offset;
	setDist<<<1, 1>>>(nodesD, SRC_NODE, 0);
	cudaDeviceSynchronize();
	checkForErr("Initialist Source");
	printf("No of edges spawn from SOURCE: %dx%d\n",
			 BLOCKS(spawnEdges), THREADS(spawnEdges));
	if(BLOCKS(spawnEdges)==0 || THREADS(spawnEdges)==0){
		printf(RED "\n\nNo edges originates from the selected node!\n\n" RESET);
		exit(EXIT_SUCCESS);
	}
	find_min_dist<<<BLOCKS(spawnEdges), THREADS(spawnEdges)>>>(
				nodesD, edgesD, nodes_n, edges_n, SRC_NODE);
	cudaDeviceSynchronize();
	checkForErr("find_min_dist");

	copyEdgesToDevice(&edgesD, &edgesH, edges_n);
	checkForErr("Copy edges to device");
	findDistD<<<BLOCKS(nodes_n), THREADS(nodes_n)>>>(
				nodesD, edgesD, nodes_n, edges_n, SRC_NODE);
	checkForErr("Find Distance from Device");

	copyToHost(&nodesD, &edgesD, &nodesH, &edgesH,
				 nodes_n+1, edges_n+nodes_n);
	checkForErr("copy data back to Host");
	printResult(nodesH, nodes_n, SRC_NODE);
	printf(GRN "END\n" RESET);
	return 0;
}