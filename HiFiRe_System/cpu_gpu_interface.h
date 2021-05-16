#ifndef __CPU_GPU_INTERFACE_H_INCLUDED__
#define __CPU_GPU_INTERFACE_H_INCLUDED__

#include "gpc.h"
#include "cpu_join.h"

//for geos
#include <fstream>
#include <iostream>
#include "geos/geom/Coordinate.h"

using namespace std;
using namespace geos::geom;

typedef struct
{
  double *xminArr;
  double *yminArr;
  double *xmaxArr;
  double *ymaxArr;
}MBR;

int ST_Intersect(int L1PolNum, int L2PolNum, int* L1VNum, int* L2VNum, int *L1VPrefixSum, int *L2VPrefixSum, 
					gpc_vertex *L1Coords, gpc_vertex *L2Coords,  MBR *L1MBR, MBR* L2MBR, 
					int numTasks, int *L1TaskId, int *L2TaskId, int *taskResult);
					  
int verifyLayer(int PolNum, int* VNum, int *VPrefixSum, gpc_vertex *vertices);

#endif
