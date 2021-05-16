#include "cpu_join.h"
#include "cpu_gpu_interface.h"
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <iostream>
#include <math.h>

using namespace std;

#define sizeLarge 15
#define sizeSmall 5
#define sizeLargeArray 21
#define sizeSmallArray 11

__global__ void wakeGPUup3(int numWake)
{	
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numWake; i += blockDim.x * gridDim.x) 
    {
		int a110 = 110;
		a110 = a110*110+666;
    }
}

void wakeGPUup1(int numWake)
{
	int blockSize = 256;
	int numBlocks = (numWake+blockSize-1)/blockSize;					
	
	wakeGPUup3 <<<numBlocks, blockSize>>> (numWake);
	
	cudaDeviceSynchronize();
	
	return;
}

//transfer from double to long long
__global__ void transferPoint3(gpc_vertex *hAcopy, gpc_vertex2 *a_copy, int L1VCount)
{	
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < L1VCount; i += blockDim.x * gridDim.x) 
    {
		a_copy[i].x = hAcopy[i].x *10000000;
		a_copy[i].y = hAcopy[i].y *10000000; 
    }
}

gpc_vertex2* transfer_pointsvalues_cuda3(int L1PolNum, int* L1VNum, int *L1VPrefixSum, gpc_vertex *hA)
{
    int lastL1PolVCount = L1VNum[L1PolNum - 1];
	int L1VCount = L1VPrefixSum[L1PolNum - 1] + lastL1PolVCount;
	
	gpc_vertex2 *a = (gpc_vertex2 *)malloc(sizeof(gpc_vertex2) * L1VCount);	 
	
	//copyin
	gpc_vertex *hAcopy;
	cudaMalloc(&hAcopy, L1VCount*sizeof(gpc_vertex));
	cudaMemcpy(hAcopy, hA, L1VCount*sizeof(gpc_vertex), cudaMemcpyHostToDevice);
	//copy
	gpc_vertex2 *a_copy;
	cudaMalloc(&a_copy, L1VCount*sizeof(gpc_vertex2));
	
	int blockSize = 1024;
	int numBlocks = (L1VCount+blockSize-1)/blockSize;					
	
	transferPoint3 <<<numBlocks, blockSize>>> (hAcopy,a_copy,L1VCount);
	
	cudaDeviceSynchronize();
	
	cudaMemcpy(a, a_copy, L1VCount*sizeof(gpc_vertex2), cudaMemcpyDeviceToHost);	
	
	cudaFree(a_copy);
	cudaFree(hAcopy);
	
	return a;
}

//for wkt dataset
__global__ void transferPoint4(Coordinate *hAcopy, gpc_vertex2 *a_copy, long L1VCount)
{	
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < L1VCount; i += blockDim.x * gridDim.x) 
    {
		a_copy[i].x = hAcopy[i].x *10000000;
		a_copy[i].y = hAcopy[i].y *10000000; 
    }
}

gpc_vertex2* transfer_pointsvalues_cuda4(int L1PolNum, int* L1VNum, long *L1VPrefixSum, Coordinate *hA)
{
	//original
    int lastL1PolVCount = L1VNum[L1PolNum - 1];
	long L1VCount = L1VPrefixSum[L1PolNum - 1] + (long)lastL1PolVCount;
	
	gpc_vertex2 *a = (gpc_vertex2 *)malloc(sizeof(gpc_vertex2) * L1VCount);	 //inside for #1
	
	//cuda data
	//copyin
	Coordinate *hAcopy;
	cudaMalloc(&hAcopy, L1VCount*sizeof(Coordinate));
	cudaMemcpy(hAcopy, hA, L1VCount*sizeof(Coordinate), cudaMemcpyHostToDevice);
	//copy
	gpc_vertex2 *a_copy;
	cudaMalloc(&a_copy, L1VCount*sizeof(gpc_vertex2));
	
	//threads part
	int blockSize = 1024;
	int numBlocks = (L1VCount+(long)blockSize-1)/blockSize;					//task index
	
	transferPoint4 <<<numBlocks, blockSize>>> (hAcopy,a_copy,L1VCount);
	
	cudaDeviceSynchronize();
	
	cudaMemcpy(a, a_copy, L1VCount*sizeof(gpc_vertex2), cudaMemcpyDeviceToHost);	
	
	cudaFree(a_copy);
	cudaFree(hAcopy);
	
	return a;
}

__global__ void transferMBR3(double *xy_copy,long long *a_copy, int tasks)
{	
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tasks; i += blockDim.x * gridDim.x) 
    {
		a_copy[i] = xy_copy[i] *10000000;
    }
}

long long* transfer_boundingbox_cuda3(int tasks, double *xy)
{	
	long long *a = (long long *)malloc(sizeof(long long) * tasks);
	
	long long *a_copy;
	cudaMalloc(&a_copy, tasks*sizeof(long long));
	
	double *xy_copy;
	cudaMalloc(&xy_copy, tasks*sizeof(double));
	cudaMemcpy(xy_copy, xy, tasks*sizeof(double), cudaMemcpyHostToDevice);
	
	int blockSize = 1024;
	int numBlocks = (tasks+blockSize-1)/blockSize;					
	
	transferMBR3 <<<numBlocks, blockSize>>> (xy_copy,a_copy,tasks);
	
	cudaDeviceSynchronize();
	
	cudaMemcpy(a, a_copy, tasks*sizeof(long long), cudaMemcpyDeviceToHost);
	
	cudaFree(a_copy);
	cudaFree(xy_copy);
	
	return a;
}

//get polysketch basic information
__global__ void tileInformation3(int tasks, int *pnpL1TaskId2, int *pnpL2TaskId2, int* L1VNum2, int* L2VNum2,
		int *numOfPartL12,int *numOfPartL22,int *lastNumL12,int *lastNumL22,int *cellsizeL12, int *cellsizeL22)
{	
	for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < tasks; i += blockDim.x * gridDim.x)
	{
		int l1PolyId = pnpL1TaskId2[i];		
		int L1PolVCount = L1VNum2[l1PolyId];  //the sum of vertices of L1 (first point and last point are same)
		int l2PolyId = pnpL2TaskId2[i];		
		int L2PolVCount = L2VNum2[l2PolyId];  
			
		int cellsize1 = sizeLarge;
		int cellsize2 = sizeLarge;
		if(L1PolVCount<400){cellsize1=sizeSmall;}
		if(L2PolVCount<400){cellsize2=sizeSmall;}
			
		cellsizeL12[i] = cellsize1;
		cellsizeL22[i] = cellsize2;
			
		int partL1 = ((L1PolVCount-1)/(cellsize1-1))+1;
		int lastNumL1sub = L1PolVCount - (cellsize1-1)*(partL1-1);
		if((partL1==0)||(partL1==1)){partL1=1;lastNumL1sub=L1PolVCount;}
		if(lastNumL1sub<=3)
		{
			partL1 = partL1-1;
			lastNumL1sub = lastNumL1sub+(cellsize1-1);
		}
		numOfPartL12[i] = partL1;
		lastNumL12[i] = lastNumL1sub;
	
		int partL2 = ((L2PolVCount-1)/(cellsize2-1))+1;
		int lastNumL2sub = L2PolVCount - (cellsize2-1)*(partL2-1);
		if((partL2==0)||(partL2==1)){partL2=1;lastNumL2sub=L2PolVCount;}
		if(lastNumL2sub<=3)
		{
			partL2 = partL2-1;
			lastNumL2sub = lastNumL2sub+(cellsize2-1);
		}
		numOfPartL22[i] = partL2;
		lastNumL22[i] = lastNumL2sub;
	}
}

//initial array to 0
__global__ void initialArray0(int tasks, int *f3)
{	
	for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < tasks; i += blockDim.x * gridDim.x)
	{
		f3[i] = 0;
	}
}

//preprocess data to calculate mbr for polysketch
__global__ void calculateMBR5(int *pnpL1TaskId2, int *L1VPrefixSum2, long long *xmax1, long long *xmin1,
			long long *ymax1, long long *ymin1, int *numOfPartL12, int *lastNumL12,
			long long *prefixPQ12,int *cellsizeL12, gpc_vertex2 *ha12)
{
	int i = blockIdx.x;
	int l1PolyId = pnpL1TaskId2[i];		 
	int partL1 = numOfPartL12[i];
	int L1Last = lastNumL12[i];
	int tempprefix1 = prefixPQ12[i];
	int cellsize1 =  cellsizeL12[i];
	int L1VPrefixtemp1 = L1VPrefixSum2[l1PolyId];
	
	for (int j = threadIdx.x; j < partL1; j += blockDim.x) 
    {
		long long xmaxtemp = ha12[L1VPrefixtemp1 + (cellsize1-1)*j ].x;
		long long xmintemp = ha12[L1VPrefixtemp1 + (cellsize1-1)*j ].x;
		long long ymaxtemp = ha12[L1VPrefixtemp1 + (cellsize1-1)*j ].y;
		long long ymintemp = ha12[L1VPrefixtemp1 + (cellsize1-1)*j ].y;
					
		int maxtemp = cellsize1;
		if (j == (partL1-1)){maxtemp = L1Last;}
				
		for(int k = 0; k<maxtemp;k++)
		{
			gpc_vertex2 aPoint = ha12[L1VPrefixtemp1+ (cellsize1-1)*j + k ]; 
			if(xmaxtemp<aPoint.x){xmaxtemp=aPoint.x;}
			if(xmintemp>aPoint.x){xmintemp=aPoint.x;}
			if(ymaxtemp<aPoint.y){ymaxtemp=aPoint.y;}
			if(ymintemp>aPoint.y){ymintemp=aPoint.y;}
		}
		xmax1[tempprefix1+j] = xmaxtemp;
		xmin1[tempprefix1+j] = xmintemp;
		ymax1[tempprefix1+j] = ymaxtemp;
		ymin1[tempprefix1+j] = ymintemp;
    }
}

//for wkt dataset
__global__ void calculateMBR7(int *pnpL1TaskId2, long *L1VPrefixSum2, long long *xmax1, long long *xmin1,long long *ymax1, long long *ymin1, 
								int *numOfPartL12, int *lastNumL12,long long *prefixPQ12,int *cellsizeL12, gpc_vertex2 *ha12)
{
	int i = blockIdx.x;
	int l1PolyId = pnpL1TaskId2[i];		 
	int partL1 = numOfPartL12[i];
	int L1Last = lastNumL12[i];
	int tempprefix1 = prefixPQ12[i];
	int cellsize1 =  cellsizeL12[i];
	long L1VPrefixtemp1 = L1VPrefixSum2[l1PolyId];
	
	for (int j = threadIdx.x; j < partL1; j += blockDim.x) 
    {
		long long xmaxtemp = ha12[L1VPrefixtemp1 + (cellsize1-1)*j ].x;
		long long xmintemp = ha12[L1VPrefixtemp1 + (cellsize1-1)*j ].x;
		long long ymaxtemp = ha12[L1VPrefixtemp1 + (cellsize1-1)*j ].y;
		long long ymintemp = ha12[L1VPrefixtemp1 + (cellsize1-1)*j ].y;
					
		int maxtemp = cellsize1;
		if (j == (partL1-1)){maxtemp = L1Last;}
				
		for(int k = 0; k<maxtemp;k++)
		{
			gpc_vertex2 aPoint = ha12[L1VPrefixtemp1+ (cellsize1-1)*j + k ]; 
			if(xmaxtemp<aPoint.x){xmaxtemp=aPoint.x;}
			if(xmintemp>aPoint.x){xmintemp=aPoint.x;}
			if(ymaxtemp<aPoint.y){ymaxtemp=aPoint.y;}
			if(ymintemp>aPoint.y){ymintemp=aPoint.y;}
		}
		xmax1[tempprefix1+j] = xmaxtemp;
		xmin1[tempprefix1+j] = xmintemp;
		ymax1[tempprefix1+j] = ymaxtemp;
		ymin1[tempprefix1+j] = ymintemp;
    }
}

//basic PSCMBR
__global__ void cudaPSketch_L21(int *pnpL1TaskId2, int *L1VPrefixSum2, long long *xmax1, long long *xmin1,long long *ymax1, 
		long long *ymin1, int *numOfPartL12, int *lastNumL12,long long *prefixPQ12,int *cellsizeL12,
		gpc_vertex2 *ha12,int *pnpL2TaskId2, int *L2VPrefixSum2, long long *xmax2, long long *xmin2,
		long long *ymax2, long long *ymin2, int *numOfPartL22, int *lastNumL22,long long *prefixPQ22,
		int *cellsizeL22, gpc_vertex2 *hb12, int *f2, int *L1Large2, gpc_vertex2 *fpoints2,int *numofPI2)
{	
	int icurrent = blockIdx.x;
	int i = L1Large2[icurrent];
	
	int partL1 = numOfPartL12[i];
	int partL2 = numOfPartL22[i];
	
	int l1PolyId = pnpL1TaskId2[i];		 
	int l2PolyId = pnpL2TaskId2[i];		 
	int L1Last = lastNumL12[i];
	int L2Last = lastNumL22[i];
	int tempprefix1 = prefixPQ12[i];
	int tempprefix2 = prefixPQ22[i];
	int cellsize1 =  cellsizeL12[i];
	int cellsize2 =  cellsizeL22[i];
	
	for (int j = threadIdx.x; j < partL1; j += blockDim.x) 
	{	
		for (int k = 0; k < partL2; k++)
		{
			if(!((xmin1[tempprefix1+j]>xmax2[tempprefix2+k])||(xmin2[tempprefix2+k]>xmax1[tempprefix1+j])||(ymax1[tempprefix1+j]<ymin2[tempprefix2+k])||(ymax2[tempprefix2+k]<ymin1[tempprefix1+j])))
			{	
				int maxtemp1 = cellsize1;
				if (j == (partL1-1)){maxtemp1 = L1Last;}
				int maxtemp2 = cellsize2;
				if (k == (partL2-1)){maxtemp2 = L2Last;}
				
				//change the array size according to the tile size!!
				//should be more than tilesize becasue the lastTileNum
				Line3 a[sizeLargeArray];
				Line3 b[sizeLargeArray];
				
				//calculate CMBR
				long long cbbxs1 = max(xmin1[tempprefix1+j],xmin2[tempprefix2+k]);
				long long cbbys1 = max(ymin1[tempprefix1+j],ymin2[tempprefix2+k]);
				long long cbbxb1 = min(xmax1[tempprefix1+j],xmax2[tempprefix2+k]);
				long long cbbyb1 = min(ymax1[tempprefix1+j],ymax2[tempprefix2+k]);
				
				int tempj1 = 0;
				int tempk1 = 0;
				
				for(int jjj=0;jjj<(maxtemp1-1);jjj++)
				{
					gpc_vertex2 test = ha12[L1VPrefixSum2[l1PolyId] + (cellsize1-1)*j + jjj ]; 
					gpc_vertex2 test2 = ha12[L1VPrefixSum2[l1PolyId] + (cellsize1-1)*j + jjj+1 ];
					
					long long cbbxs2 = test.x; 
					long long cbbys2 = test.y; 
					long long cbbxb2 = test2.x; 
					long long cbbyb2 = test2.y; 
					
					if(test.x>test2.x){cbbxs2=test2.x;cbbxb2=test.x;}
					if(test.y>test2.y){cbbys2=test2.y;cbbyb2=test.y;}
					
					if(!((cbbxs1>cbbxb2)||(cbbxs2>cbbxb1)||(cbbyb1<cbbys2)||(cbbyb2<cbbys1)))
					{
						a[tempj1].p1=test;
						a[tempj1].p2=test2;
						tempj1++;
					}
			
				}
				
				if(tempj1>0){
					for(int kkk=0;kkk<(maxtemp2-1);kkk++)
					{
						gpc_vertex2 test = hb12[L2VPrefixSum2[l2PolyId] + (cellsize2-1)*k + kkk ];
						gpc_vertex2 test2 = hb12[L2VPrefixSum2[l2PolyId] + (cellsize2-1)*k + kkk+1 ];
					
						long long cbbxs2 = test.x; 
						long long cbbys2 = test.y; 
						long long cbbxb2 = test2.x; 
						long long cbbyb2 = test2.y; 
					
						if(test.x>test2.x){cbbxs2=test2.x;cbbxb2=test.x;}
						if(test.y>test2.y){cbbys2=test2.y;cbbyb2=test.y;}
					
						if(!((cbbxs1>cbbxb2)||(cbbxs2>cbbxb1)||(cbbyb1<cbbys2)||(cbbyb2<cbbys1)))
						{
							b[tempk1].p1=test;
							b[tempk1].p2=test2;
							tempk1++;
						}
					}
				
					if(tempk1>0)
					{
						for(int jjj=0;jjj<tempj1;jjj++)
						{
							gpc_vertex2 test = a[jjj].p1; 
							gpc_vertex2 test2 = a[jjj].p2; 
				
							for(int kkk=0;kkk<tempk1;kkk++)
							{
								gpc_vertex2 vi = b[kkk].p1;
								gpc_vertex2 vj = b[kkk].p2;
									
								long long o1 = (vi.x-test.x)*(test2.y-test.y) - (vi.y-test.y)*(test2.x-test.x); 
								long long o2 = (vj.x-test.x)*(test2.y-test.y) - (vj.y-test.y)*(test2.x-test.x); 
								long long o3 = (test.x-vi.x)*(vj.y-vi.y) - (test.y-vi.y)*(vj.x-vi.x);			  
								long long o4 = (test2.x-vi.x)*(vj.y-vi.y) - (test2.y-vi.y)*(vj.x-vi.x);		 
				  
								//check intersections
								if(((o1 < 0)!= (o2 < 0))&&((o3 < 0)!= (o4 < 0)))
								{
									atomicAdd(&f2[i], 1);
						
									gpc_vertex2 P_intersection;
							
									//if(test2.x == test.x){test2.x = test2.x+1;}
									//if(vj.x == vi.x){vj.x = vj.x+1;}
										
									long long l1m = ((test2.y - test.y) / (test2.x - test.x));
									long long l1c = (test.y) - l1m*(test.x);
									long long l2m = ((vj.y - vi.y) / (vj.x - vi.x));
									long long l2c = (vi.y) - l2m*(vi.x);
										
									//if(l1m == l2m){l1m = l1m+1;}
								
									P_intersection.x = (l2c - l1c)/(l1m - l2m);
									P_intersection.y = l1m*P_intersection.x + l1c;
							
									int realNumPoints1 = atomicAdd(&numofPI2[0], 1);
									fpoints2[realNumPoints1] = P_intersection;
							
								}
								else if((o1==0)&&(vi.x<=max(test.x,test2.x))&&(vi.x>=min(test.x,test2.x))&&
										(vi.y<=max(test.y,test2.y))&&(vi.y>=min(test.y,test2.y)))
								{
									//vi on test-test2
									atomicAdd(&f2[i], 1);	
								}
								else if((o3==0)&&(test.x<=max(vi.x,vj.x))&&(test.x>=min(vi.x,vj.x))&&
											(test.y<=max(vi.y,vj.y))&&(test.y>=min(vi.y,vj.y)))
								{
									//test on vi-vj
									atomicAdd(&f2[i], 1);
								}
							}
						}
					}
				}
			}
		}		
	}
	//cudaDeviceSynchronize();
}

//group with 21
__global__ void cudaPSketch_L22(int *pnpL1TaskId2, int *L1VPrefixSum2, long long *xmax1, long long *xmin1,long long *ymax1, 
		long long *ymin1, int *numOfPartL12, int *lastNumL12,long long *prefixPQ12,int *cellsizeL12,
		gpc_vertex2 *ha12,int *pnpL2TaskId2, int *L2VPrefixSum2, long long *xmax2, long long *xmin2,
		long long *ymax2, long long *ymin2, int *numOfPartL22, int *lastNumL22,long long *prefixPQ22,
		int *cellsizeL22, gpc_vertex2 *hb12, int *f2,int *L2Large2, gpc_vertex2 *fpoints2,int *numofPI2)
{	
	int icurrent = blockIdx.x;
	int i = L2Large2[icurrent];
	
	int partL1 = numOfPartL12[i];
	int partL2 = numOfPartL22[i];
	
	int l1PolyId = pnpL1TaskId2[i];		 
	int l2PolyId = pnpL2TaskId2[i];		 
	int L1Last = lastNumL12[i];
	int L2Last = lastNumL22[i];
	int tempprefix1 = prefixPQ12[i];
	int tempprefix2 = prefixPQ22[i];
	int cellsize1 =  cellsizeL12[i];
	int cellsize2 =  cellsizeL22[i];
	
	for (int j = threadIdx.x; j < partL2; j += blockDim.x) 
	{
		for (int k = 0; k < partL1; k++)
		{
			if(!((xmin1[tempprefix1+k]>xmax2[tempprefix2+j])||(xmin2[tempprefix2+j]>xmax1[tempprefix1+k])||(ymax1[tempprefix1+k]<ymin2[tempprefix2+j])||(ymax2[tempprefix2+j]<ymin1[tempprefix1+k])))
			{	
				int maxtemp1 = cellsize2;
				if (j == (partL2-1)){maxtemp1 = L2Last;}
				int maxtemp2 = cellsize1;
				if (k == (partL1-1)){maxtemp2 = L1Last;}
				
				//change the array size according to the tile size!!
				//should be more than tilesize becasue the lastTileNum
				Line3 a[sizeLargeArray];
				Line3 b[sizeLargeArray];
				
				//calculate CMBR
				long long cbbxs1 = max(xmin1[tempprefix1+k],xmin2[tempprefix2+j]);
				long long cbbys1 = max(ymin1[tempprefix1+k],ymin2[tempprefix2+j]);
				long long cbbxb1 = min(xmax1[tempprefix1+k],xmax2[tempprefix2+j]);
				long long cbbyb1 = min(ymax1[tempprefix1+k],ymax2[tempprefix2+j]);
				
				int tempj1 = 0;
				int tempk1 = 0;
				
				for(int jjj=0;jjj<(maxtemp1-1);jjj++)
				{
					gpc_vertex2 test = hb12[L2VPrefixSum2[l2PolyId] + (cellsize2-1)*j + jjj ]; 
					gpc_vertex2 test2 = hb12[L2VPrefixSum2[l2PolyId] + (cellsize2-1)*j + jjj+1 ]; 
					
					long long cbbxs2 = test.x; 
					long long cbbys2 = test.y; 
					long long cbbxb2 = test2.x; 
					long long cbbyb2 = test2.y; 
					
					if(test.x>test2.x){cbbxs2=test2.x;cbbxb2=test.x;}
					if(test.y>test2.y){cbbys2=test2.y;cbbyb2=test.y;}
					
					if(!((cbbxs1>cbbxb2)||(cbbxs2>cbbxb1)||(cbbyb1<cbbys2)||(cbbyb2<cbbys1)))
					{
						a[tempj1].p1=test;
						a[tempj1].p2=test2;
						tempj1++;
					}

				}
				
				if(tempj1>0){
					for(int kkk=0;kkk<(maxtemp2-1);kkk++)
					{
						gpc_vertex2 test = ha12[L1VPrefixSum2[l1PolyId] + (cellsize1-1)*k + kkk ];
						gpc_vertex2 test2 = ha12[L1VPrefixSum2[l1PolyId] + (cellsize1-1)*k + kkk+1 ];
					
						long long cbbxs2 = test.x; 
						long long cbbys2 = test.y; 
						long long cbbxb2 = test2.x; 
						long long cbbyb2 = test2.y; 
					
						if(test.x>test2.x){cbbxs2=test2.x;cbbxb2=test.x;}
						if(test.y>test2.y){cbbys2=test2.y;cbbyb2=test.y;}
					
						if(!((cbbxs1>cbbxb2)||(cbbxs2>cbbxb1)||(cbbyb1<cbbys2)||(cbbyb2<cbbys1)))
						{
							b[tempk1].p1=test;
							b[tempk1].p2=test2;
							tempk1++;
						}
					}
		
					if(tempk1>0)
					{
						for(int jjj=0;jjj<tempj1;jjj++)
						{
							gpc_vertex2 test = a[jjj].p1; 
							gpc_vertex2 test2 = a[jjj].p2; 
				
							for(int kkk=0;kkk<tempk1;kkk++)
							{
								gpc_vertex2 vi = b[kkk].p1;
								gpc_vertex2 vj = b[kkk].p2;
									
								long long o1 = (vi.x-test.x)*(test2.y-test.y) - (vi.y-test.y)*(test2.x-test.x); 
								long long o2 = (vj.x-test.x)*(test2.y-test.y) - (vj.y-test.y)*(test2.x-test.x); 
								long long o3 = (test.x-vi.x)*(vj.y-vi.y) - (test.y-vi.y)*(vj.x-vi.x);			  
								long long o4 = (test2.x-vi.x)*(vj.y-vi.y) - (test2.y-vi.y)*(vj.x-vi.x);		  
				  
								//check intersections
								if(((o1 < 0)!= (o2 < 0))&&((o3 < 0)!= (o4 < 0)))
								{
									atomicAdd(&f2[i], 1);
							
									gpc_vertex2 P_intersection;
							
									//if(test2.x == test.x){test2.x = test2.x+1;}
									//if(vj.x == vi.x){vj.x = vj.x+1;}
										
									long long l1m = ((test2.y - test.y) / (test2.x - test.x));
									long long l1c = (test.y) - l1m*(test.x);
									long long l2m = ((vj.y - vi.y) / (vj.x - vi.x));
									long long l2c = (vi.y) - l2m*(vi.x);
										
									//if(l1m == l2m){l1m = l1m+1;}
							
									P_intersection.x = (l2c - l1c)/(l1m - l2m);
									P_intersection.y = l1m*P_intersection.x + l1c;
							
									int realNumPoints1 = atomicAdd(&numofPI2[0], 1);
									fpoints2[realNumPoints1] = P_intersection;
					
								}
								else if((o1==0)&&(vi.x<=max(test.x,test2.x))&&(vi.x>=min(test.x,test2.x))&&
										(vi.y<=max(test.y,test2.y))&&(vi.y>=min(test.y,test2.y)))
								{
									//vi on test-test2
									atomicAdd(&f2[i], 1);
								}
								else if((o3==0)&&(test.x<=max(vi.x,vj.x))&&(test.x>=min(vi.x,vj.x))&&
											(test.y<=max(vi.y,vj.y))&&(test.y>=min(vi.y,vj.y)))
								{
									//test on vi-vj
									atomicAdd(&f2[i], 1);
								}
							}
						}
					}
				}
			}
		}	
	}	
	//cudaDeviceSynchronize();
}

//for wkt dataset
__global__ void cudaPSketch_L31(int *pnpL1TaskId2, long *L1VPrefixSum2, long long *xmax1, long long *xmin1,long long *ymax1, 
								long long *ymin1, int *numOfPartL12, int *lastNumL12,long long *prefixPQ12,int *cellsizeL12,
								gpc_vertex2 *ha12,int *pnpL2TaskId2, long *L2VPrefixSum2, long long *xmax2, long long *xmin2,
								long long *ymax2, long long *ymin2, int *numOfPartL22, int *lastNumL22,long long *prefixPQ22,
								int *cellsizeL22, gpc_vertex2 *hb12, int *f2, int *L1Large2, gpc_vertex2 *fpoints2,int *numofPI2)
{	
	int icurrent = blockIdx.x;
	int i = L1Large2[icurrent];
	
	int partL1 = numOfPartL12[i];
	int partL2 = numOfPartL22[i];
	
	int l1PolyId = pnpL1TaskId2[i];		 
	int l2PolyId = pnpL2TaskId2[i];		 
	int L1Last = lastNumL12[i];
	int L2Last = lastNumL22[i];
	int tempprefix1 = prefixPQ12[i];
	int tempprefix2 = prefixPQ22[i];
	int cellsize1 =  cellsizeL12[i];
	int cellsize2 =  cellsizeL22[i];
	
	for (int j = threadIdx.x; j < partL1; j += blockDim.x) 
	{	
		for (int k = 0; k < partL2; k++)
		{
			if(!((xmin1[tempprefix1+j]>xmax2[tempprefix2+k])||(xmin2[tempprefix2+k]>xmax1[tempprefix1+j])||(ymax1[tempprefix1+j]<ymin2[tempprefix2+k])||(ymax2[tempprefix2+k]<ymin1[tempprefix1+j])))
			{	
				int maxtemp1 = cellsize1;
				if (j == (partL1-1)){maxtemp1 = L1Last;}
				int maxtemp2 = cellsize2;
				if (k == (partL2-1)){maxtemp2 = L2Last;}
				
				//change the array size according to the tile size!!
				//should be more than tilesize becasue the lastTileNum
				Line3 a[20];
				Line3 b[20];
				
				//calculate CMBR
				long long cbbxs1 = max(xmin1[tempprefix1+j],xmin2[tempprefix2+k]);
				long long cbbys1 = max(ymin1[tempprefix1+j],ymin2[tempprefix2+k]);
				long long cbbxb1 = min(xmax1[tempprefix1+j],xmax2[tempprefix2+k]);
				long long cbbyb1 = min(ymax1[tempprefix1+j],ymax2[tempprefix2+k]);
				
				int tempj1 = 0;
				int tempk1 = 0;
				
				for(int jjj=0;jjj<(maxtemp1-1);jjj++)
				{
					gpc_vertex2 test = ha12[L1VPrefixSum2[l1PolyId] + (cellsize1-1)*j + jjj ]; 
					gpc_vertex2 test2 = ha12[L1VPrefixSum2[l1PolyId] + (cellsize1-1)*j + jjj+1 ];
					
					long long cbbxs2 = test.x; 
					long long cbbys2 = test.y; 
					long long cbbxb2 = test2.x; 
					long long cbbyb2 = test2.y; 
					
					if(test.x>test2.x){cbbxs2=test2.x;cbbxb2=test.x;}
					if(test.y>test2.y){cbbys2=test2.y;cbbyb2=test.y;}
					
					if(!((cbbxs1>cbbxb2)||(cbbxs2>cbbxb1)||(cbbyb1<cbbys2)||(cbbyb2<cbbys1)))
					{
						a[tempj1].p1=test;
						a[tempj1].p2=test2;
						tempj1++;
					}
			
				}
				
				if(tempj1>0){
					for(int kkk=0;kkk<(maxtemp2-1);kkk++)
					{
						gpc_vertex2 test = hb12[L2VPrefixSum2[l2PolyId] + (cellsize2-1)*k + kkk ];
						gpc_vertex2 test2 = hb12[L2VPrefixSum2[l2PolyId] + (cellsize2-1)*k + kkk+1 ];
					
						long long cbbxs2 = test.x; 
						long long cbbys2 = test.y; 
						long long cbbxb2 = test2.x; 
						long long cbbyb2 = test2.y; 
					
						if(test.x>test2.x){cbbxs2=test2.x;cbbxb2=test.x;}
						if(test.y>test2.y){cbbys2=test2.y;cbbyb2=test.y;}
					
						if(!((cbbxs1>cbbxb2)||(cbbxs2>cbbxb1)||(cbbyb1<cbbys2)||(cbbyb2<cbbys1)))
						{
							b[tempk1].p1=test;
							b[tempk1].p2=test2;
							tempk1++;
						}
					}
				
					if(tempk1>0)
					{
						for(int jjj=0;jjj<tempj1;jjj++)
						{
							gpc_vertex2 test = a[jjj].p1; 
							gpc_vertex2 test2 = a[jjj].p2; 
				
							for(int kkk=0;kkk<tempk1;kkk++)
							{
								gpc_vertex2 vi = b[kkk].p1;
								gpc_vertex2 vj = b[kkk].p2;
									
								long long o1 = (vi.x-test.x)*(test2.y-test.y) - (vi.y-test.y)*(test2.x-test.x); 
								long long o2 = (vj.x-test.x)*(test2.y-test.y) - (vj.y-test.y)*(test2.x-test.x); 
								long long o3 = (test.x-vi.x)*(vj.y-vi.y) - (test.y-vi.y)*(vj.x-vi.x);			  
								long long o4 = (test2.x-vi.x)*(vj.y-vi.y) - (test2.y-vi.y)*(vj.x-vi.x);		  
				  
								//check intersections
								if(((o1 < 0)!= (o2 < 0))&&((o3 < 0)!= (o4 < 0)))
								{
									atomicAdd(&f2[i], 1);
						
									gpc_vertex2 P_intersection;
							
									//if(test2.x == test.x){test2.x = test2.x+1;}
									//if(vj.x == vi.x){vj.x = vj.x+1;}
										
									long long l1m = ((test2.y - test.y) / (test2.x - test.x));
									long long l1c = (test.y) - l1m*(test.x);
									long long l2m = ((vj.y - vi.y) / (vj.x - vi.x));
									long long l2c = (vi.y) - l2m*(vi.x);
										
									//if(l1m == l2m){l1m = l1m+1;}
								
									P_intersection.x = (l2c - l1c)/(l1m - l2m);
									P_intersection.y = l1m*P_intersection.x + l1c;
							
									int realNumPoints1 = atomicAdd(&numofPI2[0], 1);
									//atomicAdd(&numofPI2[0], 1);
									fpoints2[realNumPoints1] = P_intersection;
							
								}
								else if((o1==0)&&(vi.x<=max(test.x,test2.x))&&(vi.x>=min(test.x,test2.x))&&
										(vi.y<=max(test.y,test2.y))&&(vi.y>=min(test.y,test2.y)))
								{
									//vi on test-test2
									atomicAdd(&f2[i], 1);	
								}
								else if((o3==0)&&(test.x<=max(vi.x,vj.x))&&(test.x>=min(vi.x,vj.x))&&
											(test.y<=max(vi.y,vj.y))&&(test.y>=min(vi.y,vj.y)))
								{
									//test on vi-vj
									atomicAdd(&f2[i], 1);
								}
							}
						}
					}
				}
			}
		}		
	}
	//cudaDeviceSynchronize();
}


//group with 31
__global__ void cudaPSketch_L32(int *pnpL1TaskId2, long *L1VPrefixSum2, long long *xmax1, long long *xmin1,long long *ymax1, 
								long long *ymin1, int *numOfPartL12, int *lastNumL12,long long *prefixPQ12,int *cellsizeL12,
								gpc_vertex2 *ha12,int *pnpL2TaskId2, long *L2VPrefixSum2, long long *xmax2, long long *xmin2,
								long long *ymax2, long long *ymin2, int *numOfPartL22, int *lastNumL22,long long *prefixPQ22,
								int *cellsizeL22, gpc_vertex2 *hb12, int *f2,int *L2Large2, gpc_vertex2 *fpoints2,int *numofPI2)
{	
	int icurrent = blockIdx.x;
	int i = L2Large2[icurrent];
	
	int partL1 = numOfPartL12[i];
	int partL2 = numOfPartL22[i];
	
	int l1PolyId = pnpL1TaskId2[i];		 
	int l2PolyId = pnpL2TaskId2[i];		 
	int L1Last = lastNumL12[i];
	int L2Last = lastNumL22[i];
	int tempprefix1 = prefixPQ12[i];
	int tempprefix2 = prefixPQ22[i];
	int cellsize1 =  cellsizeL12[i];
	int cellsize2 =  cellsizeL22[i];
	
	for (int j = threadIdx.x; j < partL2; j += blockDim.x) 
	{
		for (int k = 0; k < partL1; k++)
		{
			if(!((xmin1[tempprefix1+k]>xmax2[tempprefix2+j])||(xmin2[tempprefix2+j]>xmax1[tempprefix1+k])||(ymax1[tempprefix1+k]<ymin2[tempprefix2+j])||(ymax2[tempprefix2+j]<ymin1[tempprefix1+k])))
			{	
				int maxtemp1 = cellsize2;
				if (j == (partL2-1)){maxtemp1 = L2Last;}
				int maxtemp2 = cellsize1;
				if (k == (partL1-1)){maxtemp2 = L1Last;}
				
				//change the array size according to the tile size!!
				//should be more than tilesize becasue the lastTileNum
				Line3 a[20];
				Line3 b[20];
				
				//calculate CMBR
				long long cbbxs1 = max(xmin1[tempprefix1+k],xmin2[tempprefix2+j]);
				long long cbbys1 = max(ymin1[tempprefix1+k],ymin2[tempprefix2+j]);
				long long cbbxb1 = min(xmax1[tempprefix1+k],xmax2[tempprefix2+j]);
				long long cbbyb1 = min(ymax1[tempprefix1+k],ymax2[tempprefix2+j]);
				
				int tempj1 = 0;
				int tempk1 = 0;
				
				for(int jjj=0;jjj<(maxtemp1-1);jjj++)
				{
					gpc_vertex2 test = hb12[L2VPrefixSum2[l2PolyId] + (cellsize2-1)*j + jjj ]; 
					gpc_vertex2 test2 = hb12[L2VPrefixSum2[l2PolyId] + (cellsize2-1)*j + jjj+1 ]; 
					
					long long cbbxs2 = test.x; 
					long long cbbys2 = test.y; 
					long long cbbxb2 = test2.x; 
					long long cbbyb2 = test2.y; 
					
					if(test.x>test2.x){cbbxs2=test2.x;cbbxb2=test.x;}
					if(test.y>test2.y){cbbys2=test2.y;cbbyb2=test.y;}
					
					if(!((cbbxs1>cbbxb2)||(cbbxs2>cbbxb1)||(cbbyb1<cbbys2)||(cbbyb2<cbbys1)))
					{
						a[tempj1].p1=test;
						a[tempj1].p2=test2;
						tempj1++;
					}

				}
				
				if(tempj1>0){
					for(int kkk=0;kkk<(maxtemp2-1);kkk++)
					{
						gpc_vertex2 test = ha12[L1VPrefixSum2[l1PolyId] + (cellsize1-1)*k + kkk ];
						gpc_vertex2 test2 = ha12[L1VPrefixSum2[l1PolyId] + (cellsize1-1)*k + kkk+1 ];
					
						long long cbbxs2 = test.x; 
						long long cbbys2 = test.y; 
						long long cbbxb2 = test2.x; 
						long long cbbyb2 = test2.y; 
					
						if(test.x>test2.x){cbbxs2=test2.x;cbbxb2=test.x;}
						if(test.y>test2.y){cbbys2=test2.y;cbbyb2=test.y;}
					
						if(!((cbbxs1>cbbxb2)||(cbbxs2>cbbxb1)||(cbbyb1<cbbys2)||(cbbyb2<cbbys1)))
						{
							b[tempk1].p1=test;
							b[tempk1].p2=test2;
							tempk1++;
						}
					}
		
					if(tempk1>0)
					{
						for(int jjj=0;jjj<tempj1;jjj++)
						{
							gpc_vertex2 test = a[jjj].p1; 
							gpc_vertex2 test2 = a[jjj].p2; 
				
							for(int kkk=0;kkk<tempk1;kkk++)
							{
								gpc_vertex2 vi = b[kkk].p1;
								gpc_vertex2 vj = b[kkk].p2;
									
								long long o1 = (vi.x-test.x)*(test2.y-test.y) - (vi.y-test.y)*(test2.x-test.x); 
								long long o2 = (vj.x-test.x)*(test2.y-test.y) - (vj.y-test.y)*(test2.x-test.x); 
								long long o3 = (test.x-vi.x)*(vj.y-vi.y) - (test.y-vi.y)*(vj.x-vi.x);			  
								long long o4 = (test2.x-vi.x)*(vj.y-vi.y) - (test2.y-vi.y)*(vj.x-vi.x);		  
				  
								//check intersections
								if(((o1 < 0)!= (o2 < 0))&&((o3 < 0)!= (o4 < 0)))
								{
									atomicAdd(&f2[i], 1);
							
									gpc_vertex2 P_intersection;
							
									//if(test2.x == test.x){test2.x = test2.x+1;}
									//if(vj.x == vi.x){vj.x = vj.x+1;}
										
									long long l1m = ((test2.y - test.y) / (test2.x - test.x));
									long long l1c = (test.y) - l1m*(test.x);
									long long l2m = ((vj.y - vi.y) / (vj.x - vi.x));
									long long l2c = (vi.y) - l2m*(vi.x);
										
									//if(l1m == l2m){l1m = l1m+1;}
							
									P_intersection.x = (l2c - l1c)/(l1m - l2m);
									P_intersection.y = l1m*P_intersection.x + l1c;
							
									int realNumPoints1 = atomicAdd(&numofPI2[0], 1);
									//atomicAdd(&numofPI2[0], 1);
									fpoints2[realNumPoints1] = P_intersection;
					
								}
								else if((o1==0)&&(vi.x<=max(test.x,test2.x))&&(vi.x>=min(test.x,test2.x))&&
										(vi.y<=max(test.y,test2.y))&&(vi.y>=min(test.y,test2.y)))
								{
									//vi on test-test2
									atomicAdd(&f2[i], 1);
								}
								else if((o3==0)&&(test.x<=max(vi.x,vj.x))&&(test.x>=min(vi.x,vj.x))&&
											(test.y<=max(vi.y,vj.y))&&(test.y>=min(vi.y,vj.y)))
								{
									//test on vi-vj
									atomicAdd(&f2[i], 1);
								}
							}
						}
					}
				}
			}
		}	
	}	
	//cudaDeviceSynchronize();
}

//also store information for PNP test
__global__ void cudaPSketch_L3(int *pnpL1TaskId2, int *L1VPrefixSum2, long long *xmax1, long long *xmin1,
		long long *ymax1, long long *ymin1, int *numOfPartL12, int *lastNumL12,long long *prefixPQ12,int *cellsizeL12,
		gpc_vertex2 *ha12,int *pnpL2TaskId2, int *L2VPrefixSum2, long long *xmax2, long long *xmin2,
		long long *ymax2, long long *ymin2, int *numOfPartL22, int *lastNumL22,long long *prefixPQ22,int *cellsizeL22,
		gpc_vertex2 *hb12, int *f2, int *overlaptile1, int *overlaptile2, int *L1Large2, gpc_vertex2 *fpoints2,
		int *numofPI2)
{	
	int icurrent = blockIdx.x;
	int i = L1Large2[icurrent];
	
	int partL1 = numOfPartL12[i];
	int partL2 = numOfPartL22[i];
	
	int l1PolyId = pnpL1TaskId2[i];		 
	int l2PolyId = pnpL2TaskId2[i];		 
	int L1Last = lastNumL12[i];
	int L2Last = lastNumL22[i];
	int tempprefix1 = prefixPQ12[i];
	int tempprefix2 = prefixPQ22[i];
	int cellsize1 =  cellsizeL12[i];
	int cellsize2 =  cellsizeL22[i];
	
	for (int j = threadIdx.x; j < partL1; j += blockDim.x) 
	{	
		int countr1 = 0;
		for (int k = 0; k < partL2; k++){
			if(!((xmin1[tempprefix1+j]>xmax2[tempprefix2+k])||(xmin2[tempprefix2+k]>xmax1[tempprefix1+j])||(ymax1[tempprefix1+j]<ymin2[tempprefix2+k])||(ymax2[tempprefix2+k]<ymin1[tempprefix1+j])))
			{	
				int maxtemp1 = cellsize1;
				if (j == (partL1-1)){maxtemp1 = L1Last;}
				int maxtemp2 = cellsize2;
				if (k == (partL2-1)){maxtemp2 = L2Last;}
				
				//change the array size according to the tile size!!
				//should be more than tilesize becasue the lastTileNum
				Line3 a[20];
				Line3 b[20];
				
				//calculate CMBR
				long long cbbxs1 = max(xmin1[tempprefix1+j],xmin2[tempprefix2+k]);
				long long cbbys1 = max(ymin1[tempprefix1+j],ymin2[tempprefix2+k]);
				long long cbbxb1 = min(xmax1[tempprefix1+j],xmax2[tempprefix2+k]);
				long long cbbyb1 = min(ymax1[tempprefix1+j],ymax2[tempprefix2+k]);
				
				int tempj1 = 0;
				int tempk1 = 0;
				
				for(int jjj=0;jjj<(maxtemp1-1);jjj++)
				{
					gpc_vertex2 test = ha12[L1VPrefixSum2[l1PolyId] + (cellsize1-1)*j + jjj ]; 
					gpc_vertex2 test2 = ha12[L1VPrefixSum2[l1PolyId] + (cellsize1-1)*j + jjj+1 ];
					
					long long cbbxs2 = test.x; 
					long long cbbys2 = test.y; 
					long long cbbxb2 = test2.x; 
					long long cbbyb2 = test2.y; 
					
					if(test.x>test2.x){cbbxs2=test2.x;cbbxb2=test.x;}
					if(test.y>test2.y){cbbys2=test2.y;cbbyb2=test.y;}
					
					if(!((cbbxs1>cbbxb2)||(cbbxs2>cbbxb1)||(cbbyb1<cbbys2)||(cbbyb2<cbbys1)))
					{
						a[tempj1].p1=test;
						a[tempj1].p2=test2;
						tempj1++;
					}
				}
				
				if(tempj1>0){
					for(int kkk=0;kkk<(maxtemp2-1);kkk++)
					{
						gpc_vertex2 test = hb12[L2VPrefixSum2[l2PolyId] + (cellsize2-1)*k + kkk ];
						gpc_vertex2 test2 = hb12[L2VPrefixSum2[l2PolyId] + (cellsize2-1)*k + kkk+1 ];
					
						long long cbbxs2 = test.x; 
						long long cbbys2 = test.y; 
						long long cbbxb2 = test2.x; 
						long long cbbyb2 = test2.y; 
					
						if(test.x>test2.x){cbbxs2=test2.x;cbbxb2=test.x;}
						if(test.y>test2.y){cbbys2=test2.y;cbbyb2=test.y;}
					
						if(!((cbbxs1>cbbxb2)||(cbbxs2>cbbxb1)||(cbbyb1<cbbys2)||(cbbyb2<cbbys1)))
						{
							b[tempk1].p1=test;
							b[tempk1].p2=test2;
							tempk1++;
						}
					}
				
					if(tempk1>0)
					{
						for(int jjj=0;jjj<tempj1;jjj++)
						{
							gpc_vertex2 test = a[jjj].p1; 
							gpc_vertex2 test2 = a[jjj].p2; 
				
							for(int kkk=0;kkk<tempk1;kkk++)
							{
								gpc_vertex2 vi = b[kkk].p1;
								gpc_vertex2 vj = b[kkk].p2;
									
								long long o1 = (vi.x-test.x)*(test2.y-test.y) - (vi.y-test.y)*(test2.x-test.x); 
								long long o2 = (vj.x-test.x)*(test2.y-test.y) - (vj.y-test.y)*(test2.x-test.x); 
								long long o3 = (test.x-vi.x)*(vj.y-vi.y) - (test.y-vi.y)*(vj.x-vi.x);			  
								long long o4 = (test2.x-vi.x)*(vj.y-vi.y) - (test2.y-vi.y)*(vj.x-vi.x);		  
				  
								//check intersections
								if(((o1 < 0)!= (o2 < 0))&&((o3 < 0)!= (o4 < 0)))
								{
									atomicAdd(&f2[i], 1);
									
									countr1++;
									overlaptile2[tempprefix2+k] = 1;
						
									gpc_vertex2 P_intersection;
							
									//if(test2.x == test.x){test2.x = test2.x+1;}
									//if(vj.x == vi.x){vj.x = vj.x+1;}
										
									long long l1m = ((test2.y - test.y) / (test2.x - test.x));
									long long l1c = (test.y) - l1m*(test.x);
									long long l2m = ((vj.y - vi.y) / (vj.x - vi.x));
									long long l2c = (vi.y) - l2m*(vi.x);
										
									//if(l1m == l2m){l1m = l1m+1;}
								
									P_intersection.x = (l2c - l1c)/(l1m - l2m);
									P_intersection.y = l1m*P_intersection.x + l1c;
							
									int realNumPoints1 = atomicAdd(&numofPI2[0], 1);
									//atomicAdd(&numofPI2[0], 1);
									fpoints2[realNumPoints1] = P_intersection;
							
								}
								else if((o1==0)&&(vi.x<=max(test.x,test2.x))&&(vi.x>=min(test.x,test2.x))&&
										(vi.y<=max(test.y,test2.y))&&(vi.y>=min(test.y,test2.y)))
								{
									//vi on test-test2
									atomicAdd(&f2[i], 1);	
								}
								else if((o3==0)&&(test.x<=max(vi.x,vj.x))&&(test.x>=min(vi.x,vj.x))&&
											(test.y<=max(vi.y,vj.y))&&(test.y>=min(vi.y,vj.y)))
								{
									//test on vi-vj
									atomicAdd(&f2[i], 1);
								}
							}
						}
					}
				}
			}
		}		
		if(countr1>0){
			overlaptile1[tempprefix1+j] = 1;
		}
	}
}

//group with 3
__global__ void cudaPSketch_L4(int *pnpL1TaskId2, int *L1VPrefixSum2, long long *xmax1, long long *xmin1,
		long long *ymax1, long long *ymin1, int *numOfPartL12, int *lastNumL12,long long *prefixPQ12,int *cellsizeL12,
		gpc_vertex2 *ha12,int *pnpL2TaskId2, int *L2VPrefixSum2, long long *xmax2, long long *xmin2,
		long long *ymax2, long long *ymin2, int *numOfPartL22, int *lastNumL22,long long *prefixPQ22,int *cellsizeL22,
		gpc_vertex2 *hb12, int *f2, int *overlaptile1, int *overlaptile2, int *L2Large2, gpc_vertex2 *fpoints2,
		int *numofPI2)
{	
	int icurrent = blockIdx.x;
	int i = L2Large2[icurrent];
	int partL1 = numOfPartL12[i];
	int partL2 = numOfPartL22[i];
	
	int l1PolyId = pnpL1TaskId2[i];		 
	int l2PolyId = pnpL2TaskId2[i];		 
	int L1Last = lastNumL12[i];
	int L2Last = lastNumL22[i];
	int tempprefix1 = prefixPQ12[i];
	int tempprefix2 = prefixPQ22[i];
	int cellsize1 =  cellsizeL12[i];
	int cellsize2 =  cellsizeL22[i];
	
	for (int j = threadIdx.x; j < partL2; j += blockDim.x) 
	{
		int countr2 = 0;
		for (int k = 0; k < partL1; k++){
			if(!((xmin1[tempprefix1+k]>xmax2[tempprefix2+j])||(xmin2[tempprefix2+j]>xmax1[tempprefix1+k])||(ymax1[tempprefix1+k]<ymin2[tempprefix2+j])||(ymax2[tempprefix2+j]<ymin1[tempprefix1+k])))
			{	
				int maxtemp1 = cellsize2;
				if (j == (partL2-1)){maxtemp1 = L2Last;}
				int maxtemp2 = cellsize1;
				if (k == (partL1-1)){maxtemp2 = L1Last;}

				//change the array size according to the tile size!!
				//should be more than tilesize becasue the lastTileNum
				Line3 a[20];
				Line3 b[20];
				
				//calculate CMBR
				long long cbbxs1 = max(xmin1[tempprefix1+k],xmin2[tempprefix2+j]);
				long long cbbys1 = max(ymin1[tempprefix1+k],ymin2[tempprefix2+j]);
				long long cbbxb1 = min(xmax1[tempprefix1+k],xmax2[tempprefix2+j]);
				long long cbbyb1 = min(ymax1[tempprefix1+k],ymax2[tempprefix2+j]);
				
				int tempj1 = 0;
				int tempk1 = 0;
				
				for(int jjj=0;jjj<(maxtemp1-1);jjj++)
				{
					gpc_vertex2 test = hb12[L2VPrefixSum2[l2PolyId] + (cellsize2-1)*j + jjj ]; 
					gpc_vertex2 test2 = hb12[L2VPrefixSum2[l2PolyId] + (cellsize2-1)*j + jjj+1 ]; 
					
					long long cbbxs2 = test.x; 
					long long cbbys2 = test.y; 
					long long cbbxb2 = test2.x; 
					long long cbbyb2 = test2.y; 
					
					if(test.x>test2.x){cbbxs2=test2.x;cbbxb2=test.x;}
					if(test.y>test2.y){cbbys2=test2.y;cbbyb2=test.y;}
					
					if(!((cbbxs1>cbbxb2)||(cbbxs2>cbbxb1)||(cbbyb1<cbbys2)||(cbbyb2<cbbys1)))
					{
						a[tempj1].p1=test;
						a[tempj1].p2=test2;
						tempj1++;
					}

				}
				
				if(tempj1>0){
					for(int kkk=0;kkk<(maxtemp2-1);kkk++)
					{
						gpc_vertex2 test = ha12[L1VPrefixSum2[l1PolyId] + (cellsize1-1)*k + kkk ];
						gpc_vertex2 test2 = ha12[L1VPrefixSum2[l1PolyId] + (cellsize1-1)*k + kkk+1 ];
					
						long long cbbxs2 = test.x; 
						long long cbbys2 = test.y; 
						long long cbbxb2 = test2.x; 
						long long cbbyb2 = test2.y; 
					
						if(test.x>test2.x){cbbxs2=test2.x;cbbxb2=test.x;}
						if(test.y>test2.y){cbbys2=test2.y;cbbyb2=test.y;}
					
						if(!((cbbxs1>cbbxb2)||(cbbxs2>cbbxb1)||(cbbyb1<cbbys2)||(cbbyb2<cbbys1)))
						{
							b[tempk1].p1=test;
							b[tempk1].p2=test2;
							tempk1++;
						}
					}
		
					if(tempk1>0)
					{
						for(int jjj=0;jjj<tempj1;jjj++)
						{
							gpc_vertex2 test = a[jjj].p1; 
							gpc_vertex2 test2 = a[jjj].p2; 
				
							for(int kkk=0;kkk<tempk1;kkk++)
							{
								gpc_vertex2 vi = b[kkk].p1;
								gpc_vertex2 vj = b[kkk].p2;
									
								long long o1 = (vi.x-test.x)*(test2.y-test.y) - (vi.y-test.y)*(test2.x-test.x); 
								long long o2 = (vj.x-test.x)*(test2.y-test.y) - (vj.y-test.y)*(test2.x-test.x); 
								long long o3 = (test.x-vi.x)*(vj.y-vi.y) - (test.y-vi.y)*(vj.x-vi.x);			  
								long long o4 = (test2.x-vi.x)*(vj.y-vi.y) - (test2.y-vi.y)*(vj.x-vi.x);		  
				  
								//check intersections
								if(((o1 < 0)!= (o2 < 0))&&((o3 < 0)!= (o4 < 0)))
								{
									atomicAdd(&f2[i], 1);
									countr2++;
									overlaptile1[tempprefix1+k] = 1;
							
									gpc_vertex2 P_intersection;
							
									//if(test2.x == test.x){test2.x = test2.x+1;}
									//if(vj.x == vi.x){vj.x = vj.x+1;}
										
									long long l1m = ((test2.y - test.y) / (test2.x - test.x));
									long long l1c = (test.y) - l1m*(test.x);
									long long l2m = ((vj.y - vi.y) / (vj.x - vi.x));
									long long l2c = (vi.y) - l2m*(vi.x);
										
									//if(l1m == l2m){l1m = l1m+1;}
							
									P_intersection.x = (l2c - l1c)/(l1m - l2m);
									P_intersection.y = l1m*P_intersection.x + l1c;
							
									int realNumPoints1 = atomicAdd(&numofPI2[0], 1);
									//atomicAdd(&numofPI2[0], 1);
									fpoints2[realNumPoints1] = P_intersection;
					
								}
								else if((o1==0)&&(vi.x<=max(test.x,test2.x))&&(vi.x>=min(test.x,test2.x))&&
										(vi.y<=max(test.y,test2.y))&&(vi.y>=min(test.y,test2.y)))
								{
									//vi on test-test2
									atomicAdd(&f2[i], 1);
								}
								else if((o3==0)&&(test.x<=max(vi.x,vj.x))&&(test.x>=min(vi.x,vj.x))&&
											(test.y<=max(vi.y,vj.y))&&(test.y>=min(vi.y,vj.y)))
								{
									//test on vi-vj
									atomicAdd(&f2[i], 1);
								}
							}
						}
					}
				}

			}
		}	
		if(countr2>0){
			overlaptile2[tempprefix2+j] = 1;
		}
	}
		
	//cudaDeviceSynchronize();
}

//PNP test for the tasks where one mbr is inside another mbr and no intersection points
__global__ void oneInsideAnother3(int *pnpL1TaskId2, int *L1VNum2, int *L1VPrefixSum2, 
		gpc_vertex2 *ha12,int *pnpL2TaskId2, int *L2VNum2, int *L2VPrefixSum2,
		gpc_vertex2 *hb12, int polygonInsideT, int *refineOne, int *aa1,long long *ymax2, 
		long long *ymin2, int *numOfPartL22, int *lastNumL22,
		long long *prefixPQ22,int *cellsizeL22)
{	
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < polygonInsideT; i += blockDim.x * gridDim.x) 
    {
		int l2PolyId = pnpL2TaskId2[refineOne[i]];
			
		int l1PolyId = pnpL1TaskId2[refineOne[i]];
		int L1PolVCount = L1VNum2[l1PolyId]-1;
		
		int partL2 = numOfPartL22[refineOne[i]];
		int L2Last = lastNumL22[refineOne[i]];
		int tempprefix2 = prefixPQ22[refineOne[i]];
		int cellsize2 =  cellsizeL22[refineOne[i]];
		
		char d;  // for the result of function InPoly
			
		//r means inside
		int r = 0;

		int newNNNum =  L1PolVCount;
		if(newNNNum >  10) 
		{
			newNNNum = 10;
		}
			
		for (int j = 0; j < newNNNum ; j++)    
		{   
			//a point from first polygon
			gpc_vertex2 test = ha12[L1VPrefixSum2[l1PolyId] + j];			 
				
			d='z';

			long long x = 0;
			int Rcross = 0; // number of right edge/ray crossings 
			int Lcross = 0; // number of left edge/ray crossings 
			int Zcross = 0; // For the point is a vertex of a polygon
				 				   
			for (int kk = 0; kk < partL2; kk++)
			{
				if(!((test.y<ymin2[tempprefix2+kk])||(ymax2[tempprefix2+kk]<test.y)))
				{
					int maxPNP2 = cellsize2;
					if (kk == (partL2-1)){maxPNP2 = L2Last;}
					
					for(int k4 = 0; k4 < (maxPNP2-1); k4++ ) 
					{	
						gpc_vertex2 vi = hb12[L2VPrefixSum2[l2PolyId] + (cellsize2-1)*kk + k4 ];
						gpc_vertex2 vj = hb12[L2VPrefixSum2[l2PolyId] + (cellsize2-1)*kk + k4+1 ];
						
						if ( (vi.x==test.x) && (vi.y==test.y) )
						{
							Zcross++; 
						}

						if( ( (vi.y-test.y) > 0 ) != ( (vj.y-test.y) > 0 ) ) 
						{
							x = ((vi.x-test.x) * (vj.y-test.y) - (vj.x-test.x) * (vi.y-test.y))
									/ ((vj.y-test.y) - (vi.y-test.y));
								
							if (x > 0) Rcross++;
						}

						if ( ( (vi.y-test.y) < 0 ) != ( (vj.y-test.y) < 0 ) )
						{
							x = ((vi.x-test.x) * (vj.y-test.y) - (vj.x-test.x) * (vi.y-test.y))
									/ ((vj.y-test.y) - (vi.y-test.y)); 
								
							if (x < 0) Lcross++;
						}
					}
				}
			}
			
			if(Zcross != 0 ) {d = 'v';}
			else 
			{
				if(( Rcross % 2 )!=(Lcross % 2 )){d='e';}
				else if( (Rcross % 2) == 1){d='i';r++;}
				else {d='o';}
			}
		}  
		
		if(r>3)
		{
			aa1[i] = L1PolVCount;	//inside
		}
    }
}

//group with 3
__global__ void oneInsideAnother4(int *pnpL1TaskId2, int *L1VNum2, int *L1VPrefixSum2, 
		gpc_vertex2 *ha12,int *pnpL2TaskId2, int *L2VNum2, int *L2VPrefixSum2,
		gpc_vertex2 *hb12, int polygonInsideT2, int *refineTwo, int *aa2,long long *ymax1, 
		long long *ymin1, int *numOfPartL12, int *lastNumL12,long long *prefixPQ12,int *cellsizeL12)
{	
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < polygonInsideT2; i += blockDim.x * gridDim.x) 
    {
		int l1PolyId = pnpL1TaskId2[refineTwo[i]];
			
		int l2PolyId = pnpL2TaskId2[refineTwo[i]];
		int L2PolVCount = L2VNum2[l2PolyId]-1;	
		
		int partL1 = numOfPartL12[refineTwo[i]]; 	 
		int L1Last = lastNumL12[refineTwo[i]];
		int tempprefix1 = prefixPQ12[refineTwo[i]];
		int cellsize1 =  cellsizeL12[refineTwo[i]];
		
		char d;  // for the result of function InPoly
			
		//r means inside
		int r = 0;
				
		int newNNNum =  L2PolVCount;
		if(newNNNum >  10) 
		{
			newNNNum = 10;
		}
			
		for (int j = 0; j < newNNNum; j++)    
		{   
			gpc_vertex2 test = hb12[L2VPrefixSum2[l2PolyId] + j];			 
				
			d='z';

			long long x = 0;
			int Rcross = 0; // number of right edge/ray crossings 
			int Lcross = 0; // number of left edge/ray crossings 
			int Zcross = 0; // For the point is a vertex of a polygon
				 				   		   
			for (int kk = 0; kk < partL1; kk++){
				if(!((ymax1[tempprefix1+kk]<test.y)||(test.y<ymin1[tempprefix1+kk])))
				{	
					int maxPNP2 = cellsize1;
					if (kk == (partL1-1)){maxPNP2 = L1Last;}
				
					for(int k4 = 0; k4 < (maxPNP2-1); k4++ ) 
					{	
						//choose two points from second polygon
						gpc_vertex2 vi = ha12[L1VPrefixSum2[l1PolyId] + (cellsize1-1)*kk + k4 ];
						gpc_vertex2 vj = ha12[L1VPrefixSum2[l1PolyId] + (cellsize1-1)*kk + k4+1 ];
					
						if ( (vi.x==test.x) && (vi.y==test.y) )
						{
							Zcross++; 
						}

						if( ( (vi.y-test.y) > 0 ) != ( (vj.y-test.y) > 0 ) ) 
						{
							x = ((vi.x-test.x) * (vj.y-test.y) - (vj.x-test.x) * (vi.y-test.y))
									/ ((vj.y-test.y) - (vi.y-test.y));
								
							if (x > 0) Rcross++;
						}

						if ( ( (vi.y-test.y) < 0 ) != ( (vj.y-test.y) < 0 ) )
						{
							x = ((vi.x-test.x) * (vj.y-test.y) - (vj.x-test.x) * (vi.y-test.y))
									/ ((vj.y-test.y) - (vi.y-test.y)); 
								
							if (x < 0) Lcross++;
						}	
					}
				}
			}
						
			if(Zcross != 0 ) {d = 'v';}
			else 
			{
				if(( Rcross % 2 )!=(Lcross % 2 )){d='e';}
				else if( (Rcross % 2) == 1){d='i';r++;}
				else {d='o';}
			}
		}  
		
		if(r>3)
		{
			aa2[i] = L2PolVCount;	//inside
		}
    }
}

__global__ void finPNP4(int *pnpL1TaskId2, int *L1VPrefixSum2, long long *ymax1, long long *ymin1, int *numOfPartL12, 
		int *lastNumL12,long long *prefixPQ12,int *cellsizeL12, gpc_vertex2 *ha12,int *pnpL2TaskId2,  
		int *L2VPrefixSum2, long long *ymax2, long long *ymin2, int *numOfPartL22, int *lastNumL22,
		long long *prefixPQ22,int *cellsizeL22, gpc_vertex2 *hb12, int *overlaptile2, int *newNum2, int *f4)
{	
	int iCurrent=blockIdx.x;
	int i = newNum2[iCurrent];
	int partL1 = numOfPartL12[i];
	int partL2 = numOfPartL22[i];
	
	int l1PolyId = pnpL1TaskId2[i];		 
	int l2PolyId = pnpL2TaskId2[i];		 
	int L1Last = lastNumL12[i];
	int L2Last = lastNumL22[i];
	int tempprefix1 = prefixPQ12[i];
	int tempprefix2 = prefixPQ22[i];
	int cellsize1 =  cellsizeL12[i];
	int cellsize2 =  cellsizeL22[i];
	
	for (int j = threadIdx.x; j < partL2; j += blockDim.x) 
	{	
		int r = 0;
		
		int maxtemp2 = cellsize2;
		if (j == (partL2-1)){maxtemp2 = L2Last;}
			
		if(overlaptile2[tempprefix2+j] == 1)
		{
			for(int jj=0; jj<(maxtemp2-1);jj++)
			{
				gpc_vertex2 test4 = hb12[L2VPrefixSum2[l2PolyId] + (cellsize2-1)*j + jj];
					
				char d='z';

				long long x = 0;
				int Rcross = 0; // number of right edge/ray crossings 
				int Lcross = 0; // number of left edge/ray crossings 
				int Zcross = 0; // For the point is a vertex of a polygon
		
				for (int kk = 0; kk < partL1; kk++){
					if(!((ymax1[tempprefix1+kk]<test4.y)||(test4.y<ymin1[tempprefix1+kk])))
					{	
						int maxPNP2 = cellsize1;
						if (kk == (partL1-1)){maxPNP2 = L1Last;}
				
						for(int k4 = 0; k4 < (maxPNP2-1); k4++ ) 
						{	
							gpc_vertex2 vi = ha12[L1VPrefixSum2[l1PolyId] + (cellsize1-1)*kk + k4 ];
							gpc_vertex2 vj = ha12[L1VPrefixSum2[l1PolyId] + (cellsize1-1)*kk + k4+1 ];
					
							if ( (vi.x==test4.x) && (vi.y==test4.y) )
							{
								Zcross++;  
							}

							if( ( (vi.y-test4.y) > 0 ) != ( (vj.y-test4.y) > 0 ) ) 
							{
								x = ((vi.x-test4.x) * (vj.y-test4.y) - (vj.x-test4.x) * (vi.y-test4.y))
										/ ((vj.y-test4.y) - (vi.y-test4.y));
								
								if (x > 0) Rcross++;
							}

							if ( ( (vi.y-test4.y) < 0 ) != ( (vj.y-test4.y) < 0 ) )
							{
								x = ((vi.x-test4.x) * (vj.y-test4.y) - (vj.x-test4.x) * (vi.y-test4.y))
										/ ((vj.y-test4.y) - (vi.y-test4.y)); 
								
								if (x < 0) Lcross++;
							}	
						}
					}
				}		
				if(Zcross != 0 ) {d = 'v';}
				else 
				{
					if(( Rcross % 2 )!=(Lcross % 2 )){d='e';}
					else if( (Rcross % 2) == 1){d='i'; r++; }
					else {d='o';}
				}
			}
		}
			
		else if(overlaptile2[tempprefix2+j] == 0)
		{
			int j4 =1;
			gpc_vertex2 test4 = hb12[L2VPrefixSum2[l2PolyId] + (cellsize2-1)*j + j4];
					
			char d='z';

			long long x = 0;
			int Rcross = 0; // number of right edge/ray crossings 
			int Lcross = 0; // number of left edge/ray crossings 
			int Zcross = 0; // For the point is a vertex of a polygon
		
			for (int kk = 0; kk < partL1; kk ++){
				if(!((ymax1[tempprefix1+kk]<test4.y)||(test4.y<ymin1[tempprefix1+kk])))
				{	
					int maxPNP2 = cellsize1;
					if (kk == (partL1-1)){maxPNP2 = L1Last;}
				
					for(int k4 = 0; k4 < (maxPNP2-1); k4++ ) 
					{	
						gpc_vertex2 vi = ha12[L1VPrefixSum2[l1PolyId] + (cellsize1-1)*kk + k4 ];
						gpc_vertex2 vj = ha12[L1VPrefixSum2[l1PolyId] + (cellsize1-1)*kk + k4+1 ];
					
						if ( (vi.x==test4.x) && (vi.y==test4.y) )
						{
							Zcross++; 
						}

						if( ( (vi.y-test4.y) > 0 ) != ( (vj.y-test4.y) > 0 ) ) 
						{
							x = ((vi.x-test4.x) * (vj.y-test4.y) - (vj.x-test4.x) * (vi.y-test4.y))
									/ ((vj.y-test4.y) - (vi.y-test4.y));
							
							if (x > 0) Rcross++;
						}

						if ( ( (vi.y-test4.y) < 0 ) != ( (vj.y-test4.y) < 0 ) )
						{
							x = ((vi.x-test4.x) * (vj.y-test4.y) - (vj.x-test4.x) * (vi.y-test4.y))
									/ ((vj.y-test4.y) - (vi.y-test4.y)); 
								
							if (x < 0) Lcross++;
						}	
					}
				}
			}		
			if(Zcross != 0 ) {d = 'v';}
			else 
			{
				if(( Rcross % 2 )!=(Lcross % 2 )){d='e';}
				else if( (Rcross % 2) == 1){d='i'; r =r+maxtemp2-1; }
				else {d='o';}
			}
		}
		atomicAdd(&f4[iCurrent], r);
	}
	//cudaDeviceSynchronize();
}

//group with 4	
__global__ void finPNP5(int *pnpL1TaskId2, int *L1VPrefixSum2, long long *ymax1, long long *ymin1, int *numOfPartL12, 
		int *lastNumL12,long long *prefixPQ12,int *cellsizeL12, gpc_vertex2 *ha12,int *pnpL2TaskId2, 
		int *L2VPrefixSum2, long long *ymax2, long long *ymin2, int *numOfPartL22, int *lastNumL22,
		long long *prefixPQ22,int *cellsizeL22, gpc_vertex2 *hb12, int *overlaptile1, int *newNum2,int *f6)
{	
	int iCurrent=blockIdx.x;
	int i = newNum2[iCurrent];
	int partL1 = numOfPartL12[i];
	int partL2 = numOfPartL22[i];
	
	int l1PolyId = pnpL1TaskId2[i];		 
	int l2PolyId = pnpL2TaskId2[i];		 
	int L1Last = lastNumL12[i];
	int L2Last = lastNumL22[i];
	int tempprefix1 = prefixPQ12[i];
	int tempprefix2 = prefixPQ22[i];
	int cellsize1 =  cellsizeL12[i];
	int cellsize2 =  cellsizeL22[i];
	
	for (int j = threadIdx.x; j < partL1; j += blockDim.x) 
	{
		int r = 0;
		
		int maxtemp1 = cellsize1;
		if (j == (partL1-1)){maxtemp1 = L1Last;}
			
		if(overlaptile1[tempprefix1+j] == 1)
		{
			for(int jj=0; jj<(maxtemp1-1);jj++)
			{
				gpc_vertex2 test4 = ha12[L1VPrefixSum2[l1PolyId] + (cellsize1-1)*j + jj];			 
			
				char d='z';

				long long x = 0;
				int Rcross = 0; // number of right edge/ray crossings 
				int Lcross = 0; // number of left edge/ray crossings 
				int Zcross = 0; // For the point is a vertex of a polygon
		
				for (int kk = 0; kk < partL2; kk ++){
					if(!((test4.y<ymin2[tempprefix2+kk])||(ymax2[tempprefix2+kk]<test4.y)))
					{	
						int maxPNP2 = cellsize2;
						if (kk == (partL2-1)){maxPNP2 = L2Last;}
				
						for(int k4 = 0; k4 < (maxPNP2-1); k4++ ) 
						{	
							gpc_vertex2 vi = hb12[L2VPrefixSum2[l2PolyId] + (cellsize2-1)*kk + k4 ];
							gpc_vertex2 vj = hb12[L2VPrefixSum2[l2PolyId] + (cellsize2-1)*kk + k4+1 ];
					
							if ( (vi.x==test4.x) && (vi.y==test4.y) )
							{
								Zcross++; 
							}

							if( ( (vi.y-test4.y) > 0 ) != ( (vj.y-test4.y) > 0 ) ) 
							{
								x = ((vi.x-test4.x) * (vj.y-test4.y) - (vj.x-test4.x) * (vi.y-test4.y))
										/ ((vj.y-test4.y) - (vi.y-test4.y));
								
								if (x > 0) Rcross++;
							}

							if ( ( (vi.y-test4.y) < 0 ) != ( (vj.y-test4.y) < 0 ) )
							{
								x = ((vi.x-test4.x) * (vj.y-test4.y) - (vj.x-test4.x) * (vi.y-test4.y))
										/ ((vj.y-test4.y) - (vi.y-test4.y)); 
								
								if (x < 0) Lcross++;
							}	
						}
					}
				}		
				if(Zcross != 0 ) {d = 'v';}
				else 
				{
					if(( Rcross % 2 )!=(Lcross % 2 )){d='e';}
					else if( (Rcross % 2) == 1){d='i'; r++;}
					else {d='o';}
				}
			}
		}
			
		else if(overlaptile1[tempprefix1+j] == 0)
		{
			int jj=1;
			gpc_vertex2 test4 = ha12[L1VPrefixSum2[l1PolyId] + (cellsize1-1)*j + jj];			 
				
			char d='z';

			long long x = 0;
			int Rcross = 0; // number of right edge/ray crossings 
			int Lcross = 0; // number of left edge/ray crossings 
			int Zcross = 0; // For the point is a vertex of a polygon
		
			for (int kk = 0; kk < partL2; kk ++){
				if(!((test4.y<ymin2[tempprefix2+kk])||(ymax2[tempprefix2+kk]<test4.y)))
				{	
					int maxPNP2 = cellsize2;
					if (kk == (partL2-1)){maxPNP2 = L2Last;}
				
					for(int k4 = 0; k4 < (maxPNP2-1); k4++ ) 
					{	
						gpc_vertex2 vi = hb12[L2VPrefixSum2[l2PolyId] + (cellsize2-1)*kk + k4 ];
						gpc_vertex2 vj = hb12[L2VPrefixSum2[l2PolyId] + (cellsize2-1)*kk + k4+1 ];
					
						if ( (vi.x==test4.x) && (vi.y==test4.y) )
						{
							Zcross++;  
						}

						if( ( (vi.y-test4.y) > 0 ) != ( (vj.y-test4.y) > 0 ) ) 
						{
							x = ((vi.x-test4.x) * (vj.y-test4.y) - (vj.x-test4.x) * (vi.y-test4.y))
									/ ((vj.y-test4.y) - (vi.y-test4.y));
								
							if (x > 0) Rcross++;
						}

						if ( ( (vi.y-test4.y) < 0 ) != ( (vj.y-test4.y) < 0 ) )
						{
							x = ((vi.x-test4.x) * (vj.y-test4.y) - (vj.x-test4.x) * (vi.y-test4.y))
									/ ((vj.y-test4.y) - (vi.y-test4.y)); 
								
							if (x < 0) Lcross++;
						}	
					}
				}
			}		
			if(Zcross != 0 ) {d = 'v';}
			else 
			{
				if(( Rcross % 2 )!=(Lcross % 2 )){d='e';}
				else if( (Rcross % 2) == 1){d='i'; r = r+maxtemp1-1;}
				else {d='o';}
			}
		}
		atomicAdd(&f6[iCurrent], r);
	}
		
	//cudaDeviceSynchronize();
}

//PolySketch-CMBR
int* pSketch23(int tasks, int *pnpL1TaskId, int *pnpL2TaskId, int L1PolNum, int L2PolNum, int* L1VNum, int* L2VNum,
		int *L1VPrefixSum, int *L2VPrefixSum, gpc_vertex2 *ha1, gpc_vertex2 *hb1,long long* rect1n,long long* rect2n,
		long long* rect3n,long long* rect4n,long long* rect1_queryn, long long* rect2_queryn, long long* rect3_queryn,
		long long* rect4_queryn)
{	
	double starttime1, endtime1;
    double difference1;	
    starttime1 = my_difftime();

	int lastL1PolVCount = L1VNum[L1PolNum - 1];
	int L1VCount = L1VPrefixSum[L1PolNum - 1] + lastL1PolVCount;
	int lastL2PolVCount = L2VNum[L2PolNum - 1];
	int L2VCount = L2VPrefixSum[L2PolNum - 1] + lastL2PolVCount;
	
	//copyin 
	int *pnpL1TaskId2, *pnpL2TaskId2, *L1VNum2, *L2VNum2, *L1VPrefixSum2, *L2VPrefixSum2;
	gpc_vertex2 *ha12, *hb12;
	
	//copyin
	cudaMalloc(&pnpL1TaskId2, tasks*sizeof(int));
	cudaMalloc(&pnpL2TaskId2, tasks*sizeof(int));
	cudaMalloc(&L1VNum2, L1PolNum*sizeof(int));
	cudaMalloc(&L2VNum2, L2PolNum*sizeof(int));
	cudaMalloc(&L1VPrefixSum2, L1PolNum*sizeof(int));
	cudaMalloc(&L2VPrefixSum2, L2PolNum*sizeof(int));
	cudaMalloc(&ha12, L1VCount*sizeof(gpc_vertex2));
	cudaMalloc(&hb12, L2VCount*sizeof(gpc_vertex2));
	
	//copyin
	cudaMemcpy(pnpL1TaskId2, pnpL1TaskId, tasks*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(pnpL2TaskId2, pnpL2TaskId, tasks*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(L1VNum2, L1VNum, L1PolNum*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(L2VNum2, L2VNum, L2PolNum*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(L1VPrefixSum2, L1VPrefixSum, L1PolNum*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(L2VPrefixSum2, L2VPrefixSum, L2PolNum*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(ha12, ha1, L1VCount*sizeof(gpc_vertex2), cudaMemcpyHostToDevice);
	cudaMemcpy(hb12, hb1, L2VCount*sizeof(gpc_vertex2), cudaMemcpyHostToDevice);
	
	//device 
	int *numOfPartL12, *numOfPartL22, *lastNumL12, *lastNumL22,*cellsizeL12, *cellsizeL22;
	long long *prefixPQ12, *prefixPQ22;
	
	//device
	cudaMalloc(&numOfPartL12, tasks*sizeof(int));
	cudaMalloc(&numOfPartL22, tasks*sizeof(int));
	cudaMalloc(&lastNumL12, tasks*sizeof(int));
	cudaMalloc(&lastNumL22, tasks*sizeof(int));
	cudaMalloc(&cellsizeL12, tasks*sizeof(int));
	cudaMalloc(&cellsizeL22, tasks*sizeof(int));
	cudaMalloc(&prefixPQ12, tasks*sizeof(long long));
	cudaMalloc(&prefixPQ22, tasks*sizeof(long long));
	
	//stream
	cudaStream_t tempf0,tempf1,tempf2,tempf31, tempf32;
	cudaStreamCreateWithFlags(&tempf0, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&tempf1, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&tempf2, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&tempf31, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&tempf32, cudaStreamNonBlocking);
	
	int blockSize0 = 1024;
	int numBlocks0 = (tasks+blockSize0-1)/blockSize0;
	
	//get sketch basic information
	tileInformation3 <<<numBlocks0, blockSize0,0,tempf0>>> (tasks, pnpL1TaskId2, pnpL2TaskId2, L1VNum2, L2VNum2,
							numOfPartL12, numOfPartL22, lastNumL12, lastNumL22, cellsizeL12, cellsizeL22);
					
	cudaDeviceSynchronize();
	
	int *numOfPartL1 = (int *)malloc(tasks * sizeof(int));   //how many parts of 1st polygon
	int *numOfPartL2 = (int *)malloc(tasks * sizeof(int)); 
	
	cudaMemcpy(numOfPartL1, numOfPartL12, tasks*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(numOfPartL2, numOfPartL22, tasks*sizeof(int), cudaMemcpyDeviceToHost);
	
	int *L1Large = (int *)malloc(tasks * sizeof(int)); 
	int *L2Large = (int *)malloc(tasks * sizeof(int)); 
	int L1LargeNum =0;
	int L2LargeNum =0;

	long long *prefixPQ1 = (long long *)malloc(tasks * sizeof(long long)); 	//the number of numOfPartL1 
	long long *prefixPQ2 = (long long *)malloc(tasks * sizeof(long long)); 
	
	prefixPQ1[0] = 0;
	prefixPQ2[0] = 0;
	
	//assign to the polygon has more tiles
	if(numOfPartL1[0]>=numOfPartL2[0]){L1Large[0]=0;L1LargeNum++;}
	if(numOfPartL1[0]<numOfPartL2[0]){L2Large[0]=0;L2LargeNum++;}
	
	for(int i =1;i<tasks;i++)
	{	
		prefixPQ1[i] = prefixPQ1[i-1]+ numOfPartL1[i-1];
		prefixPQ2[i] = prefixPQ2[i-1]+ numOfPartL2[i-1];
		
		if(numOfPartL1[i]>=numOfPartL2[i]){L1Large[L1LargeNum]=i;L1LargeNum++;}
		else{L2Large[L2LargeNum]=i;L2LargeNum++;}
	}
	
	cudaMemcpy(prefixPQ12, prefixPQ1, tasks*sizeof(long long), cudaMemcpyHostToDevice);
	cudaMemcpy(prefixPQ22, prefixPQ2, tasks*sizeof(long long), cudaMemcpyHostToDevice);
	
	//device
	int *L1Large2, *L2Large2;
	
	cudaMalloc(&L1Large2, L1LargeNum*sizeof(int));
	cudaMalloc(&L2Large2, L2LargeNum*sizeof(int));	
	
	cudaMemcpy(L1Large2, L1Large, L1LargeNum*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(L2Large2, L2Large, L2LargeNum*sizeof(int), cudaMemcpyHostToDevice);
	
	long long tasknum1 = prefixPQ1[tasks-1]+ numOfPartL1[tasks-1];
	long long tasknum2 = prefixPQ2[tasks-1]+ numOfPartL2[tasks-1];
	
	//local value
	int *f1 = (int *)malloc(sizeof(int) * tasks);
	gpc_vertex2 *fpoints1 = (gpc_vertex2 *)malloc(sizeof(gpc_vertex2) *3*tasks);
	int *numofPI1 = (int *)malloc(sizeof(int) * 1);
	
	//create
	long long *xmax1, *xmin1, *xmax2, *xmin2;
	long long *ymax1, *ymin1, *ymax2, *ymin2;
	
	//create
	cudaMalloc(&xmax1, tasknum1*sizeof(long long));
	cudaMalloc(&xmin1, tasknum1*sizeof(long long));
	cudaMalloc(&ymax1, tasknum1*sizeof(long long));
	cudaMalloc(&ymin1, tasknum1*sizeof(long long));
	cudaMalloc(&xmax2, tasknum2*sizeof(long long));
	cudaMalloc(&xmin2, tasknum2*sizeof(long long));
	cudaMalloc(&ymax2, tasknum2*sizeof(long long));
	cudaMalloc(&ymin2, tasknum2*sizeof(long long));
	
	//copy
	int *f2;
	gpc_vertex2 *fpoints2;	
	int *numofPI2;
	
	//copy
	cudaMalloc(&f2,tasks*sizeof(int));
	cudaMalloc(&fpoints2,3*tasks*sizeof(gpc_vertex2));
	cudaMalloc(&numofPI2,1*sizeof(int));
		
	calculateMBR5 <<<tasks, 64,0,tempf1>>> (pnpL1TaskId2, L1VPrefixSum2, xmax1, xmin1, ymax1, ymin1, numOfPartL12, lastNumL12, prefixPQ12, cellsizeL12, ha12);
	
	calculateMBR5 <<<tasks, 64,0,tempf2>>> (pnpL2TaskId2, L2VPrefixSum2, xmax2, xmin2, ymax2, ymin2, numOfPartL22, lastNumL22, prefixPQ22, cellsizeL22, hb12);
	
	cudaDeviceSynchronize();
	
	//store line segments overlap with CMBR version
	cudaPSketch_L21 <<<L1LargeNum, 256,0, tempf31>>> (pnpL1TaskId2, L1VPrefixSum2, xmax1, xmin1, ymax1, ymin1, numOfPartL12, 
													lastNumL12, prefixPQ12, cellsizeL12, ha12, pnpL2TaskId2, L2VPrefixSum2, 
													xmax2, xmin2, ymax2, ymin2, numOfPartL22, lastNumL22, prefixPQ22, cellsizeL22, 
													hb12,f2,L1Large2,fpoints2,numofPI2);
													
	cudaPSketch_L22 <<<L2LargeNum, 256,0, tempf32>>> (pnpL1TaskId2, L1VPrefixSum2, xmax1, xmin1, ymax1, ymin1, numOfPartL12, 
													lastNumL12, prefixPQ12, cellsizeL12, ha12, pnpL2TaskId2, L2VPrefixSum2, 
													xmax2, xmin2, ymax2, ymin2, numOfPartL22, lastNumL22, prefixPQ22, cellsizeL22, 
													hb12,f2,L2Large2,fpoints2,numofPI2);
	
	cudaDeviceSynchronize();
	
	//copy data from gpu to cpu
	cudaMemcpy(f1, f2, tasks*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(fpoints1, fpoints2, 3*tasks*sizeof(gpc_vertex2), cudaMemcpyDeviceToHost);
	cudaMemcpy(numofPI1, numofPI2, 1*sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();
	endtime1 = my_difftime();
	
	difference1 = endtime1 - starttime1;
	printf("total time taken =  %f\t \n",difference1);
	
	cudaFree(L1Large2);
	cudaFree(L2Large2);
	cudaFree(xmax1);
	cudaFree(xmin1);
	cudaFree(xmax2);
	cudaFree(xmin2);
	cudaFree(f2);
	cudaFree(fpoints2);
	cudaFree(numofPI2);
	
	cudaStreamDestroy(tempf0);
	cudaStreamDestroy(tempf1);
	cudaStreamDestroy(tempf2);
	cudaStreamDestroy(tempf31);
	cudaStreamDestroy(tempf32);
	
	//free memory
	cudaFree(ymax1);
	cudaFree(ymin1);
	cudaFree(ymax2);
	cudaFree(ymin2);
	
	cudaFree(pnpL1TaskId2);
	cudaFree(pnpL2TaskId2);
	cudaFree(L1VPrefixSum2);
	cudaFree(L2VPrefixSum2);
	cudaFree(ha12);
	cudaFree(hb12);
	
	cudaFree(numOfPartL12);
	cudaFree(numOfPartL22);
	cudaFree(lastNumL12);
	cudaFree(lastNumL22);
	cudaFree(cellsizeL12);
	cudaFree(cellsizeL22);
	cudaFree(prefixPQ12);
	cudaFree(prefixPQ22);
	
	return f1;
}


//PolySketch-CMBR
//for wkt dataset
int* pSketch25(int tasks, int *pnpL1TaskId, int *pnpL2TaskId, int L1PolNum, int L2PolNum, int* L1VNum, int* L2VNum,
				long *L1VPrefixSum, long *L2VPrefixSum, gpc_vertex2 *ha1, gpc_vertex2 *hb1,long long* rect1n,long long* rect2n,
				long long* rect3n,long long* rect4n,long long* rect1_queryn, long long* rect2_queryn, long long* rect3_queryn,
				long long* rect4_queryn)
{	
	//total time taken
	double starttime1, endtime1;
    double difference1;	
    starttime1 = my_difftime();
			
	//original
	int lastL1PolVCount = L1VNum[L1PolNum - 1];
	long L1VCount = L1VPrefixSum[L1PolNum - 1] + (long)lastL1PolVCount;
	int lastL2PolVCount = L2VNum[L2PolNum - 1];
	long L2VCount = L2VPrefixSum[L2PolNum - 1] + (long)lastL2PolVCount;
	
	//copyin 
	int *pnpL1TaskId2, *pnpL2TaskId2, *L1VNum2, *L2VNum2;
	long *L1VPrefixSum2, *L2VPrefixSum2;
	gpc_vertex2 *ha12, *hb12;
	
	//copyin
	cudaMalloc(&pnpL1TaskId2, tasks*sizeof(int));
	cudaMalloc(&pnpL2TaskId2, tasks*sizeof(int));
	cudaMalloc(&L1VNum2, L1PolNum*sizeof(int));
	cudaMalloc(&L2VNum2, L2PolNum*sizeof(int));
	cudaMalloc(&L1VPrefixSum2, L1PolNum*sizeof(long));
	cudaMalloc(&L2VPrefixSum2, L2PolNum*sizeof(long));
	cudaMalloc(&ha12, L1VCount*sizeof(gpc_vertex2));
	cudaMalloc(&hb12, L2VCount*sizeof(gpc_vertex2));
	
	//copyin
	cudaMemcpy(pnpL1TaskId2, pnpL1TaskId, tasks*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(pnpL2TaskId2, pnpL2TaskId, tasks*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(L1VNum2, L1VNum, L1PolNum*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(L2VNum2, L2VNum, L2PolNum*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(L1VPrefixSum2, L1VPrefixSum, L1PolNum*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(L2VPrefixSum2, L2VPrefixSum, L2PolNum*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(ha12, ha1, L1VCount*sizeof(gpc_vertex2), cudaMemcpyHostToDevice);
	cudaMemcpy(hb12, hb1, L2VCount*sizeof(gpc_vertex2), cudaMemcpyHostToDevice);
	
	//device 
	int *numOfPartL12, *numOfPartL22, *lastNumL12, *lastNumL22,*cellsizeL12, *cellsizeL22;
	long long *prefixPQ12, *prefixPQ22;
	
	//device
	cudaMalloc(&numOfPartL12, tasks*sizeof(int));
	cudaMalloc(&numOfPartL22, tasks*sizeof(int));
	cudaMalloc(&lastNumL12, tasks*sizeof(int));
	cudaMalloc(&lastNumL22, tasks*sizeof(int));
	cudaMalloc(&cellsizeL12, tasks*sizeof(int));
	cudaMalloc(&cellsizeL22, tasks*sizeof(int));
	cudaMalloc(&prefixPQ12, tasks*sizeof(long long));
	cudaMalloc(&prefixPQ22, tasks*sizeof(long long));
	
	//stream
	cudaStream_t tempf0,tempf1,tempf2,tempf31, tempf32;
	cudaStreamCreateWithFlags(&tempf0, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&tempf1, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&tempf2, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&tempf31, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&tempf32, cudaStreamNonBlocking);
	
	int blockSize0 = 1024;
	int numBlocks0 = (tasks+blockSize0-1)/blockSize0;
	
	//get sketch basic information
	tileInformation3 <<<numBlocks0, blockSize0,0,tempf0>>> (tasks, pnpL1TaskId2, pnpL2TaskId2, L1VNum2, L2VNum2,
							numOfPartL12, numOfPartL22, lastNumL12, lastNumL22, cellsizeL12, cellsizeL22);
					
	cudaDeviceSynchronize();
	
	int *numOfPartL1 = (int *)malloc(tasks * sizeof(int));   
	int *numOfPartL2 = (int *)malloc(tasks * sizeof(int)); 
	
	cudaMemcpy(numOfPartL1, numOfPartL12, tasks*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(numOfPartL2, numOfPartL22, tasks*sizeof(int), cudaMemcpyDeviceToHost);
	
	int *L1Large = (int *)malloc(tasks * sizeof(int)); 
	int *L2Large = (int *)malloc(tasks * sizeof(int)); 
	int L1LargeNum =0;
	int L2LargeNum =0;

	long long *prefixPQ1 = (long long *)malloc(tasks * sizeof(long long)); 	
	long long *prefixPQ2 = (long long *)malloc(tasks * sizeof(long long)); 
	
	prefixPQ1[0] = 0;
	prefixPQ2[0] = 0;
	
	//assign to the polygon has more tiles
	if(numOfPartL1[0]>=numOfPartL2[0]){L1Large[0]=0;L1LargeNum++;}
	if(numOfPartL1[0]<numOfPartL2[0]){L2Large[0]=0;L2LargeNum++;}
	
	for(int i =1;i<tasks;i++)
	{	
		prefixPQ1[i] = prefixPQ1[i-1]+ numOfPartL1[i-1];
		prefixPQ2[i] = prefixPQ2[i-1]+ numOfPartL2[i-1];
		
		if(numOfPartL1[i]>=numOfPartL2[i]){L1Large[L1LargeNum]=i;L1LargeNum++;}
		else{L2Large[L2LargeNum]=i;L2LargeNum++;}
	}
	
	cudaMemcpy(prefixPQ12, prefixPQ1, tasks*sizeof(long long), cudaMemcpyHostToDevice);
	cudaMemcpy(prefixPQ22, prefixPQ2, tasks*sizeof(long long), cudaMemcpyHostToDevice);
	
	//device
	int *L1Large2, *L2Large2;
	
	cudaMalloc(&L1Large2, L1LargeNum*sizeof(int));
	cudaMalloc(&L2Large2, L2LargeNum*sizeof(int));	
	
	cudaMemcpy(L1Large2, L1Large, L1LargeNum*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(L2Large2, L2Large, L2LargeNum*sizeof(int), cudaMemcpyHostToDevice);
	
	long long tasknum1 = prefixPQ1[tasks-1]+ numOfPartL1[tasks-1];
	long long tasknum2 = prefixPQ2[tasks-1]+ numOfPartL2[tasks-1];
	
	//local value
	int *f1 = (int *)malloc(sizeof(int) * tasks);
	gpc_vertex2 *fpoints1 = (gpc_vertex2 *)malloc(sizeof(gpc_vertex2) *10*tasks);
	int *numofPI1 = (int *)malloc(sizeof(int) * 1);
	
	//create
	long long *xmax1, *xmin1, *xmax2, *xmin2;
	long long *ymax1, *ymin1, *ymax2, *ymin2;
	
	//create
	cudaMalloc(&xmax1, tasknum1*sizeof(long long));
	cudaMalloc(&xmin1, tasknum1*sizeof(long long));
	cudaMalloc(&ymax1, tasknum1*sizeof(long long));
	cudaMalloc(&ymin1, tasknum1*sizeof(long long));
	cudaMalloc(&xmax2, tasknum2*sizeof(long long));
	cudaMalloc(&xmin2, tasknum2*sizeof(long long));
	cudaMalloc(&ymax2, tasknum2*sizeof(long long));
	cudaMalloc(&ymin2, tasknum2*sizeof(long long));
	
	//copy 
	int *f2;
	gpc_vertex2 *fpoints2;	
	int *numofPI2;
	
	//copy
	cudaMalloc(&f2,tasks*sizeof(int));
	cudaMalloc(&fpoints2,10*tasks*sizeof(gpc_vertex2));
	cudaMalloc(&numofPI2,1*sizeof(int));
		
	calculateMBR7 <<<tasks, 64,0,tempf1>>> (pnpL1TaskId2, L1VPrefixSum2, xmax1, xmin1, ymax1, ymin1, numOfPartL12, lastNumL12, prefixPQ12, cellsizeL12, ha12);
	
	calculateMBR7 <<<tasks, 64,0,tempf2>>> (pnpL2TaskId2, L2VPrefixSum2, xmax2, xmin2, ymax2, ymin2, numOfPartL22, lastNumL22, prefixPQ22, cellsizeL22, hb12);
	
	cudaDeviceSynchronize();
	
	//store line segments overlap with CMBR
	//for wkt dataset
	cudaPSketch_L31 <<<L1LargeNum, 256,0, tempf31>>> (pnpL1TaskId2, L1VPrefixSum2, xmax1, xmin1, ymax1, ymin1, numOfPartL12, 
													lastNumL12, prefixPQ12, cellsizeL12, ha12, pnpL2TaskId2, L2VPrefixSum2, 
													xmax2, xmin2, ymax2, ymin2, numOfPartL22, lastNumL22, prefixPQ22, cellsizeL22, 
													hb12,f2,L1Large2,fpoints2,numofPI2);
													
	cudaPSketch_L32 <<<L2LargeNum, 256,0, tempf32>>> (pnpL1TaskId2, L1VPrefixSum2, xmax1, xmin1, ymax1, ymin1, numOfPartL12, 
													lastNumL12, prefixPQ12, cellsizeL12, ha12, pnpL2TaskId2, L2VPrefixSum2, 
													xmax2, xmin2, ymax2, ymin2, numOfPartL22, lastNumL22, prefixPQ22, cellsizeL22, 
													hb12,f2,L2Large2,fpoints2,numofPI2);
	
	cudaDeviceSynchronize();
	
	//copy data from gpu to cpu
	cudaMemcpy(f1, f2, tasks*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(fpoints1, fpoints2, 10*tasks*sizeof(gpc_vertex2), cudaMemcpyDeviceToHost);
	cudaMemcpy(numofPI1, numofPI2, 1*sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();
	endtime1 = my_difftime();
	
	difference1 = endtime1 - starttime1;
	printf("total time taken =  %f\t \n",difference1);
	
	cudaFree(L1Large2);
	cudaFree(L2Large2);
	cudaFree(xmax1);
	cudaFree(xmin1);
	cudaFree(xmax2);
	cudaFree(xmin2);
	cudaFree(f2);
	cudaFree(fpoints2);
	cudaFree(numofPI2);
	
	cudaStreamDestroy(tempf0);
	cudaStreamDestroy(tempf1);
	cudaStreamDestroy(tempf2);
	cudaStreamDestroy(tempf31);
	cudaStreamDestroy(tempf32);
	
	//free memory
	cudaFree(ymax1);
	cudaFree(ymin1);
	cudaFree(ymax2);
	cudaFree(ymin2);
	
	cudaFree(pnpL1TaskId2);
	cudaFree(pnpL2TaskId2);
	cudaFree(L1VPrefixSum2);
	cudaFree(L2VPrefixSum2);
	cudaFree(ha12);
	cudaFree(hb12);
	
	cudaFree(numOfPartL12);
	cudaFree(numOfPartL22);
	cudaFree(lastNumL12);
	cudaFree(lastNumL22);
	cudaFree(cellsizeL12);
	cudaFree(cellsizeL22);
	cudaFree(prefixPQ12);
	cudaFree(prefixPQ22);
	
	return f1;
}

//store intersection points, and PNP test
int* pSketch26(int tasks, int *pnpL1TaskId, int *pnpL2TaskId, int L1PolNum, int L2PolNum, int* L1VNum, int* L2VNum,
		int *L1VPrefixSum, int *L2VPrefixSum, gpc_vertex2 *ha1, gpc_vertex2 *hb1,long long* rect1n,long long* rect2n,
		long long* rect3n,long long* rect4n,long long* rect1_queryn, long long* rect2_queryn, long long* rect3_queryn,
		long long* rect4_queryn)
{	
	//total time taken
	double starttime1, endtime1;
    double difference1;	
    starttime1 = my_difftime();
			
	//original
	int lastL1PolVCount = L1VNum[L1PolNum - 1];
	int L1VCount = L1VPrefixSum[L1PolNum - 1] + lastL1PolVCount;
	int lastL2PolVCount = L2VNum[L2PolNum - 1];
	int L2VCount = L2VPrefixSum[L2PolNum - 1] + lastL2PolVCount;
	
	//copyin 
	int *pnpL1TaskId2, *pnpL2TaskId2, *L1VNum2, *L2VNum2, *L1VPrefixSum2, *L2VPrefixSum2;
	gpc_vertex2 *ha12, *hb12;
	
	//copyin
	cudaMalloc(&pnpL1TaskId2, tasks*sizeof(int));
	cudaMalloc(&pnpL2TaskId2, tasks*sizeof(int));
	cudaMalloc(&L1VNum2, L1PolNum*sizeof(int));
	cudaMalloc(&L2VNum2, L2PolNum*sizeof(int));
	cudaMalloc(&L1VPrefixSum2, L1PolNum*sizeof(int));
	cudaMalloc(&L2VPrefixSum2, L2PolNum*sizeof(int));
	cudaMalloc(&ha12, L1VCount*sizeof(gpc_vertex2));
	cudaMalloc(&hb12, L2VCount*sizeof(gpc_vertex2));
	
	//copyin
	cudaMemcpy(pnpL1TaskId2, pnpL1TaskId, tasks*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(pnpL2TaskId2, pnpL2TaskId, tasks*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(L1VNum2, L1VNum, L1PolNum*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(L2VNum2, L2VNum, L2PolNum*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(L1VPrefixSum2, L1VPrefixSum, L1PolNum*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(L2VPrefixSum2, L2VPrefixSum, L2PolNum*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(ha12, ha1, L1VCount*sizeof(gpc_vertex2), cudaMemcpyHostToDevice);
	cudaMemcpy(hb12, hb1, L2VCount*sizeof(gpc_vertex2), cudaMemcpyHostToDevice);
	
	//device 
	int *numOfPartL12, *numOfPartL22, *lastNumL12, *lastNumL22,*cellsizeL12, *cellsizeL22;
	long long *prefixPQ12, *prefixPQ22;
	
	//device
	cudaMalloc(&numOfPartL12, tasks*sizeof(int));
	cudaMalloc(&numOfPartL22, tasks*sizeof(int));
	cudaMalloc(&lastNumL12, tasks*sizeof(int));
	cudaMalloc(&lastNumL22, tasks*sizeof(int));
	cudaMalloc(&cellsizeL12, tasks*sizeof(int));
	cudaMalloc(&cellsizeL22, tasks*sizeof(int));
	cudaMalloc(&prefixPQ12, tasks*sizeof(long long));
	cudaMalloc(&prefixPQ22, tasks*sizeof(long long));
	
	//stream
	cudaStream_t tempf0,tempf1,tempf2,tempf31, tempf32,tempf4,tempf5,tempf6,tempf7,tempf81,tempf82;
	cudaStreamCreateWithFlags(&tempf0, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&tempf1, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&tempf2, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&tempf31, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&tempf32, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&tempf4, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&tempf5, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&tempf6, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&tempf7, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&tempf81, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&tempf82, cudaStreamNonBlocking);
	
	int blockSize0 = 1024;
	int numBlocks0 = (tasks+blockSize0-1)/blockSize0;
	
	//get sketch basic information
	tileInformation3 <<<numBlocks0, blockSize0,0,tempf0>>> (tasks, pnpL1TaskId2, pnpL2TaskId2, L1VNum2, L2VNum2,
							numOfPartL12, numOfPartL22, lastNumL12, lastNumL22, cellsizeL12, cellsizeL22);
					
	cudaDeviceSynchronize();
	
	int *numOfPartL1 = (int *)malloc(tasks * sizeof(int));   //how many parts of 1st polygon
	int *numOfPartL2 = (int *)malloc(tasks * sizeof(int)); 
	
	cudaMemcpy(numOfPartL1, numOfPartL12, tasks*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(numOfPartL2, numOfPartL22, tasks*sizeof(int), cudaMemcpyDeviceToHost);
	
	int *L1Large = (int *)malloc(tasks * sizeof(int)); 
	int *L2Large = (int *)malloc(tasks * sizeof(int)); 
	int L1LargeNum =0;
	int L2LargeNum =0;

	long long *prefixPQ1 = (long long *)malloc(tasks * sizeof(long long)); 	//the number of numOfPartL1 
	long long *prefixPQ2 = (long long *)malloc(tasks * sizeof(long long)); 
	
	prefixPQ1[0] = 0;
	prefixPQ2[0] = 0;
	
	if(numOfPartL1[0]>=numOfPartL2[0]){L1Large[0]=0;L1LargeNum++;}
	if(numOfPartL1[0]<numOfPartL2[0]){L2Large[0]=0;L2LargeNum++;}
	
	for(int i =1;i<tasks;i++)
	{	
		prefixPQ1[i] = prefixPQ1[i-1]+ numOfPartL1[i-1];
		prefixPQ2[i] = prefixPQ2[i-1]+ numOfPartL2[i-1];
		
		if(numOfPartL1[i]>=numOfPartL2[i]){L1Large[L1LargeNum]=i;L1LargeNum++;}
		else{L2Large[L2LargeNum]=i;L2LargeNum++;}
	}
	
	cudaMemcpy(prefixPQ12, prefixPQ1, tasks*sizeof(long long), cudaMemcpyHostToDevice);
	cudaMemcpy(prefixPQ22, prefixPQ2, tasks*sizeof(long long), cudaMemcpyHostToDevice);
	
	//device
	int *L1Large2, *L2Large2;
	
	cudaMalloc(&L1Large2, L1LargeNum*sizeof(int));
	cudaMalloc(&L2Large2, L2LargeNum*sizeof(int));	
	
	cudaMemcpy(L1Large2, L1Large, L1LargeNum*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(L2Large2, L2Large, L2LargeNum*sizeof(int), cudaMemcpyHostToDevice);
	
	long long tasknum1 = prefixPQ1[tasks-1]+ numOfPartL1[tasks-1];
	long long tasknum2 = prefixPQ2[tasks-1]+ numOfPartL2[tasks-1];
	
	//local value
	int *f1 = (int *)malloc(sizeof(int) * tasks);
	gpc_vertex2 *fpoints1 = (gpc_vertex2 *)malloc(sizeof(gpc_vertex2) *3*tasks);
	int *numofPI1 = (int *)malloc(sizeof(int) * 1);
	
	//create
	long long *xmax1, *xmin1, *xmax2, *xmin2;
	long long *ymax1, *ymin1, *ymax2, *ymin2;
	
	int *overlaptile1, *overlaptile2;
	
	//create
	cudaMalloc(&xmax1, tasknum1*sizeof(long long));
	cudaMalloc(&xmin1, tasknum1*sizeof(long long));
	cudaMalloc(&ymax1, tasknum1*sizeof(long long));
	cudaMalloc(&ymin1, tasknum1*sizeof(long long));
	cudaMalloc(&xmax2, tasknum2*sizeof(long long));
	cudaMalloc(&xmin2, tasknum2*sizeof(long long));
	cudaMalloc(&ymax2, tasknum2*sizeof(long long));
	cudaMalloc(&ymin2, tasknum2*sizeof(long long));
	
	cudaMalloc(&overlaptile1, tasknum1*sizeof(int));
	cudaMalloc(&overlaptile2, tasknum2*sizeof(int));
	
	//copy 
	int *f2;
	gpc_vertex2 *fpoints2;	
	int *numofPI2;
	
	//copy
	cudaMalloc(&f2,tasks*sizeof(int));
	cudaMalloc(&fpoints2,3*tasks*sizeof(gpc_vertex2));
	cudaMalloc(&numofPI2,1*sizeof(int));
		
	calculateMBR5 <<<tasks, 64,0,tempf1>>> (pnpL1TaskId2, L1VPrefixSum2, xmax1, xmin1, ymax1, ymin1, numOfPartL12, lastNumL12, prefixPQ12, cellsizeL12, ha12);
	
	calculateMBR5 <<<tasks, 64,0,tempf2>>> (pnpL2TaskId2, L2VPrefixSum2, xmax2, xmin2, ymax2, ymin2, numOfPartL22, lastNumL22, prefixPQ22, cellsizeL22, hb12);
	
	cudaDeviceSynchronize();
	
	//PSCMBR and store the information for PNP test
	cudaPSketch_L3 <<<L1LargeNum, 256,0, tempf31>>> (pnpL1TaskId2, L1VPrefixSum2, xmax1, xmin1, ymax1, ymin1, numOfPartL12, 
													lastNumL12, prefixPQ12, cellsizeL12, ha12, pnpL2TaskId2, L2VPrefixSum2, 
													xmax2, xmin2, ymax2, ymin2, numOfPartL22, lastNumL22, prefixPQ22, cellsizeL22, 
													hb12,f2,overlaptile1,overlaptile2,L1Large2,fpoints2,numofPI2);
													
	cudaPSketch_L4 <<<L2LargeNum, 256,0, tempf32>>> (pnpL1TaskId2, L1VPrefixSum2, xmax1, xmin1, ymax1, ymin1, numOfPartL12, 
													lastNumL12, prefixPQ12, cellsizeL12, ha12, pnpL2TaskId2, L2VPrefixSum2, 
													xmax2, xmin2, ymax2, ymin2, numOfPartL22, lastNumL22, prefixPQ22, cellsizeL22, 
													hb12,f2,overlaptile1,overlaptile2,L2Large2,fpoints2,numofPI2);
		
	cudaDeviceSynchronize();
	
	//copy data from gpu to cpu
	cudaMemcpy(f1, f2, tasks*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(fpoints1, fpoints2, 3*tasks*sizeof(gpc_vertex2), cudaMemcpyDeviceToHost);
	cudaMemcpy(numofPI1, numofPI2, 1*sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();
	
	cudaFree(L1Large2);
	cudaFree(L2Large2);
	cudaFree(xmax1);
	cudaFree(xmin1);
	cudaFree(xmax2);
	cudaFree(xmin2);
	cudaFree(f2);
	cudaFree(fpoints2);
	cudaFree(numofPI2);
	
	cudaStreamDestroy(tempf0);
	cudaStreamDestroy(tempf1);
	cudaStreamDestroy(tempf2);
	cudaStreamDestroy(tempf31);
	cudaStreamDestroy(tempf32);
		
	int nNum = 0;  
	int *newNum = (int *)malloc(tasks * sizeof(int));  
	int polygonInsideT = 0;
	int *refineOne = (int *)malloc(tasks * sizeof(int));
	int polygonInsideT2 = 0;
	int *refineTwo = (int *)malloc(tasks * sizeof(int));
	int dDiscard = 0 ;
		
	for(int i =0; i < tasks; i++) 
	{	
		int l1PolyId = pnpL1TaskId[i];		
		int l2PolyId = pnpL2TaskId[i];
	
		if(f1[i] != 0){
			newNum[nNum] = i;
			nNum++;
		}	
		else if((rect1n[l1PolyId]>=rect1_queryn[l2PolyId])&&(rect2n[l1PolyId]>=rect2_queryn[l2PolyId])&&(rect3n[l1PolyId]<=rect3_queryn[l2PolyId])&&(rect4n[l1PolyId]<=rect4_queryn[l2PolyId]))
		{		
			//layer 1 may be inside layer 2 
			refineOne[polygonInsideT] = i;
			polygonInsideT++;  
		}
		else if((rect1n[l1PolyId]<=rect1_queryn[l2PolyId])&&(rect2n[l1PolyId]<=rect2_queryn[l2PolyId])&&(rect3n[l1PolyId]>=rect3_queryn[l2PolyId])&&(rect4n[l1PolyId]>=rect4_queryn[l2PolyId]))
		{
			//layer 2 may be inside layer 1
			refineTwo[polygonInsideT2] = i;
			polygonInsideT2++;   
		}
		else{dDiscard++;}
	}
	
	//create 
	int *refineOne2, *refineTwo2, *newNum2;
	int *aa1, *aa2;
	
	cudaMalloc(&refineOne2, polygonInsideT*sizeof(int));
	cudaMalloc(&refineTwo2, polygonInsideT2*sizeof(int));
	cudaMalloc(&newNum2, nNum*sizeof(int));
	
	cudaMemcpy(refineOne2, refineOne, polygonInsideT*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(refineTwo2, refineTwo, polygonInsideT2*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(newNum2, newNum, nNum*sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMalloc(&aa1, polygonInsideT*sizeof(int));
	cudaMalloc(&aa2, polygonInsideT2*sizeof(int));
	
	int blockSize9 = 128;
	int numBlocks91 = (polygonInsideT+blockSize9-1)/blockSize9;
	initialArray0 <<<numBlocks91, blockSize9,0,tempf81>>> (polygonInsideT, aa1);
	int numBlocks92 = (polygonInsideT2+blockSize9-1)/blockSize9;
	initialArray0 <<<numBlocks92, blockSize9,0,tempf82>>> (polygonInsideT2, aa2);
	
	cudaDeviceSynchronize();
	
	int blockSize1 = 256;
	int numBlocks1 = (polygonInsideT+blockSize1-1)/blockSize1;
				
	//layer 1 may be inside layer 2 
	oneInsideAnother3 <<<numBlocks1, blockSize1,0,tempf4>>> (pnpL1TaskId2, L1VNum2, L1VPrefixSum2, ha12,pnpL2TaskId2,
															L2VNum2, L2VPrefixSum2, hb12, polygonInsideT, refineOne2,aa1,
															ymax2, ymin2, numOfPartL22, lastNumL22, prefixPQ22, cellsizeL22);
						
	int numBlocks2 = (polygonInsideT2+blockSize1-1)/blockSize1;
	
	//layer 2 may be inside layer 1

	oneInsideAnother4 <<<numBlocks2, blockSize1,0,tempf5>>> (pnpL1TaskId2, L1VNum2, L1VPrefixSum2, ha12,pnpL2TaskId2,
															L2VNum2, L2VPrefixSum2, hb12, polygonInsideT2, refineTwo2,aa2,
															ymax1, ymin1, numOfPartL12,lastNumL12,prefixPQ12,cellsizeL12);
	
	cudaDeviceSynchronize();
	
	cudaFree(L1VNum2);
	cudaFree(L2VNum2);
	
	int *aa12 = (int *)malloc(polygonInsideT * sizeof(int));
	int *aa22 = (int *)malloc(polygonInsideT2 * sizeof(int));
	
	cudaMemcpy(aa12, aa1, polygonInsideT*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(aa22, aa2, polygonInsideT2*sizeof(int), cudaMemcpyDeviceToHost);
	
	int *f3 = (int *)malloc(sizeof(int) * nNum);
	int *f5 = (int *)malloc(sizeof(int) * nNum);
	
	//create 
	int *f4,*f6;
	cudaMalloc(&f4,nNum*sizeof(int));
	cudaMalloc(&f6,nNum*sizeof(int));
	
	int blockSize3 = 128;
	int numBlocks3 = (nNum+blockSize3-1)/blockSize3;
	
	initialArray0 <<<numBlocks3, blockSize3,0,tempf81>>> (nNum, f4);
	initialArray0 <<<numBlocks3, blockSize3,0,tempf82>>> (nNum, f6);
	
	cudaDeviceSynchronize();
	
	finPNP4 <<<nNum, 256,0, tempf6>>> (pnpL1TaskId2,  L1VPrefixSum2, ymax1, ymin1, numOfPartL12, 
													lastNumL12, prefixPQ12, cellsizeL12, ha12, pnpL2TaskId2,  L2VPrefixSum2, 
													ymax2, ymin2, numOfPartL22, lastNumL22, prefixPQ22, cellsizeL22, 
													hb12, overlaptile2, newNum2,f4);
	
	finPNP5 <<<nNum, 256,0, tempf7>>> (pnpL1TaskId2, L1VPrefixSum2, ymax1, ymin1, numOfPartL12, 
													lastNumL12, prefixPQ12, cellsizeL12, ha12, pnpL2TaskId2,  L2VPrefixSum2, 
													ymax2, ymin2, numOfPartL22, lastNumL22, prefixPQ22, cellsizeL22, 
													hb12, overlaptile1,newNum2,f6);
	
	cudaDeviceSynchronize();
	
	cudaMemcpy(f3, f4, nNum*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(f5, f6, nNum*sizeof(int), cudaMemcpyDeviceToHost);
	
	//cudaDeviceSynchronize();
	
	endtime1 = my_difftime();
	difference1 = endtime1 - starttime1;
	printf("total time taken =  %f\t \n",difference1);
	
	//free memory
	cudaFree(ymax1);
	cudaFree(ymin1);
	cudaFree(ymax2);
	cudaFree(ymin2);
	
	cudaFree(overlaptile1);
	cudaFree(overlaptile2);
	
	cudaFree(pnpL1TaskId2);
	cudaFree(pnpL2TaskId2);
	cudaFree(L1VPrefixSum2);
	cudaFree(L2VPrefixSum2);
	cudaFree(ha12);
	cudaFree(hb12);
	
	cudaFree(numOfPartL12);
	cudaFree(numOfPartL22);
	cudaFree(lastNumL12);
	cudaFree(lastNumL22);
	cudaFree(cellsizeL12);
	cudaFree(cellsizeL22);
	cudaFree(prefixPQ12);
	cudaFree(prefixPQ22);
	
	cudaFree(refineOne2);
	cudaFree(refineTwo2);
	cudaFree(newNum2);
	cudaFree(aa1);
	cudaFree(aa2);
	cudaFree(f4);
	cudaFree(f6);
	
	cudaStreamDestroy(tempf4);
	cudaStreamDestroy(tempf5);
	cudaStreamDestroy(tempf6);
	cudaStreamDestroy(tempf7);
	
	return f1;
}

