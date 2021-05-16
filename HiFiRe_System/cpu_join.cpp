#include "cpu_join.h"
#include <fstream>
#include <list>
#include <iostream>
#include <omp.h>
#include <cstring> 	
#include <string> 	
#include <math.h>
#include <mutex>
#include <thread>
#include <cstdlib>
#include <cmath>

#include "geos/geom/Geometry.h"
#include "geos/io/WKTReader.h"
#include "geos/geom/Coordinate.h"
#include "geos/geom/Envelope.h"
#include "geos/geom/LineString.h"
#include "geos/geom/Point.h"
#include "geos/geom/Polygon.h"

using namespace std;
using namespace geos::geom;

gpc_vertex2* transfer_pointsvalues_cuda3(int L1PolNum, int* L1VNum, int *L1VPrefixSum, gpc_vertex *hA);

gpc_vertex2* transfer_pointsvalues_cuda4(int L1PolNum, int* L1VNum, long *L1VPrefixSum, Coordinate *hA);

long long* transfer_boundingbox_cuda3(int tasks, double *xy);

void wakeGPUup1(int numWake);

//PolySKetch-CMBR
int* pSketch23(int tasks, int *pnpL1TaskId, int *pnpL2TaskId, int L1PolNum, int L2PolNum, int* L1VNum, int* L2VNum,
		int *L1VPrefixSum, int *L2VPrefixSum, gpc_vertex2 *ha1, gpc_vertex2 *hb1,long long* rect1n,long long* rect2n,
		long long* rect3n,long long* rect4n,long long* rect1_queryn, long long* rect2_queryn, long long* rect3_queryn,
		long long* rect4_queryn);	
					  			
//PolySKetch-CMBR for wkt dataset
int* pSketch25(int tasks, int *pnpL1TaskId, int *pnpL2TaskId, int L1PolNum, int L2PolNum, int* L1VNum, int* L2VNum,
                      long *L1VPrefixSum, long *L2VPrefixSum, gpc_vertex2 *ha1, gpc_vertex2 *hb1,long long* rect1n,long long* rect2n,
					  long long* rect3n,long long* rect4n,long long* rect1_queryn, long long* rect2_queryn, long long* rect3_queryn,
					  long long* rect4_queryn);	
					  
//PolySKetch-CMBR and PNP 
int* pSketch26(int tasks, int *pnpL1TaskId, int *pnpL2TaskId, int L1PolNum, int L2PolNum, int* L1VNum, int* L2VNum,
		int *L1VPrefixSum, int *L2VPrefixSum, gpc_vertex2 *ha1, gpc_vertex2 *hb1,long long* rect1n,long long* rect2n,
		long long* rect3n,long long* rect4n,long long* rect1_queryn, long long* rect2_queryn, long long* rect3_queryn,
		long long* rect4_queryn);	

//for geos data sets
geos::io::WKTReader wktreader;
gpc_overlayPoly2 *subjectPoly2 = NULL; 
gpc_polygon2 *clipPoly2 = NULL;

std::mutex push_mutex;
//the class is for layer(for geos)
typedef struct
{
int polNum;
vector<int> * vNum;  
vector<long> * prefixSum; 
vector<Coordinate> * coord; 
vector<Coordinate> * mbr; 
vector<Envelope> * mbr2; 
}polygonLayerTmp;

/* base layer polygons */
gpc_overlayPoly *subjectPoly = NULL; 

int    num_elementsSubPoly1;
int    num_allocatedSubPoly = 0; 

/* overlay layer polygons */
gpc_polygon *clipPoly = NULL;
 
int    num_elementsClipPoly = 0;
int    num_allocatedClipPoly = 0; 
int    totalMyHit = 0;

int SHPReadMBR1( SHPHandle psSHP, int startIndex, int endIndex, PolyRect ** mbrs){
	PolyRect * bounding_boxes;
	int num=(endIndex-startIndex+1);
	bounding_boxes=(PolyRect *) malloc(num*sizeof(PolyRect));
	int i,j;

/* -------------------------------------------------------------------- */
/*      Read the record.                                                */
/* -------------------------------------------------------------------- */
    for(i=startIndex;i<=endIndex;i++){
		if( psSHP->sHooks.FSeek( psSHP->fpSHP, psSHP->panRecOffset[i]+12, 0 ) != 0 )
		{
			char str[128];
			sprintf( str,
					 "Error in fseek() reading object from .shp file at offset %u",
					 psSHP->panRecOffset[i]+12);

			psSHP->sHooks.Error( str );
			return -1;
		}
		for(j=0;j<4;j++){
			if( psSHP->sHooks.FRead( &(bounding_boxes[i-startIndex].mbr.boundary[j]), sizeof(double), 1, psSHP->fpSHP ) != 1 )
			{
				char str[128];
				sprintf( str,
						 "Error in fread() reading object of size %u at offset %u from .shp file",
						 4*sizeof(double), psSHP->panRecOffset[i]+12 );

				psSHP->sHooks.Error( str );
				return -1;
			}			
		}
		bounding_boxes[i-startIndex].poly_id=i;
    }
    (*mbrs)=bounding_boxes;
    return 0;
}

void convert(double * rect1, double * rect2, double * rect3, double * rect4,SHPObject **psShape_base, 
		int *num_base, int * prefix_base,bBox *baseBoxes,int cellsPerProcess,int sum_mbrs_overlay,
		SHPHandle hSHP_base,double * minX, double *minY)
{
	int i;
	int prefix=0;
	
	for(i =0 ; i< baseBoxes[0].count;i++)
	{     
		rect1[i+prefix]=baseBoxes[0].rects[i].mbr.boundary[0];
		rect2[i+prefix]=baseBoxes[0].rects[i].mbr.boundary[1];
		rect3[i+prefix]=baseBoxes[0].rects[i].mbr.boundary[2];
		rect4[i+prefix]=baseBoxes[0].rects[i].mbr.boundary[3];
		if(rect1[i+prefix]<(*minX))
		{
			(*minX)=rect1[i+prefix];
		}
		if(rect2[i+prefix]<(*minY))
		{
			(*minY)=rect2[i+prefix];
		}
		psShape_base[i+prefix] = SHPReadObject( hSHP_base, baseBoxes[0].rects[i].poly_id);
		if( psShape_base[i+prefix] == NULL )
		{
			fprintf( stderr,"Unable to read shape %d, terminating object reading.\n",baseBoxes[0].rects[i].poly_id );
			exit(1);
		}
		num_base[i+prefix]=psShape_base[i+prefix]->nVertices;
		if((i+prefix)==0)
		{
			prefix_base[i+prefix]=0;
		}	
		else
		{
			prefix_base[i+prefix]=prefix_base[i+prefix-1]+num_base[i+prefix-1];
		}
	}
	prefix+=baseBoxes[0].count;
}

void destoryObjects(SHPObject **psShape,int num){
	int i;
	for(i=0;i<num;i++){
	    SHPDestroyObject(psShape[i]);
	}
}

double my_difftime ()
{
    struct timeval tp;
    struct timezone tzp;
    int i;

    i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

int MySearchCallback(int id, void* arg) 
 {
       int* idOfbase = (int *)arg;
	
       if(subjectPoly[*idOfbase].num_elementsBasePoly == subjectPoly[*idOfbase].num_allocatedBasePoly)
       {
			if (subjectPoly[*idOfbase].num_allocatedBasePoly == 0)
            subjectPoly[*idOfbase].num_allocatedBasePoly = 100; 
			else
			{
				subjectPoly[*idOfbase].num_allocatedBasePoly += 100; 
			}
			void *_tmp = realloc(subjectPoly[*idOfbase].overlayPolyIndices, (subjectPoly[*idOfbase].num_allocatedBasePoly 
			* sizeof(int)));
			if (!_tmp)
			{
				printf("************ ERROR: Couldn't realloc memory!***********\n");
				fflush(stdout);
			}
			subjectPoly[*idOfbase].overlayPolyIndices = (int *)_tmp;   
       }
	   subjectPoly[*idOfbase].overlayPolyIndices[subjectPoly[*idOfbase].num_elementsBasePoly] = id-1;
       subjectPoly[*idOfbase].num_elementsBasePoly++;
       return 1;
}

void rtreeBuildingAndSearch(int num_contours, double *rect1, double *rect2, double *rect3,double *rect4, 
		int *id_base, int numOfQuerys,double *rect1_query,double *rect2_query, double *rect3_query, double *rect4_query)
{
   subjectPoly = (gpc_overlayPoly *)malloc(num_contours * 
			  sizeof(gpc_overlayPoly));
  
    struct Node* root = RTreeNewIndex();
    int nhits,i;
			  
	for(i=0; i<numOfQuerys; i++)
	{
	    struct Rect rect = {rect1_query[i], rect2_query[i], rect3_query[i], rect4_query[i]};
        RTreeInsertRect(&rect, i+1, &root, 0);
    }
	
	int c = 0;
	for (c= 0; c < num_contours; c++)
    {
      gpc_overlayPoly p;
      p.num_elementsBasePoly = 0;
      p.num_allocatedBasePoly = 0;
      p.overlayPolyIndices = NULL;
      subjectPoly[c] = p;
	}
					
 	int a;
	for(a = 0; a <num_contours; a++)
    {
	  nhits = 0;
	  struct Rect search_rect = {rect1[a], rect2[a], rect3[a], rect4[a]};
	  nhits = RTreeSearch(root, &search_rect, MySearchCallback, &a);
	}
}


//function for geos
polygonLayerTmp* populateLayerData(list<Geometry*>* geoms);
void readGeomsFromStr(vector<string> *vstr, list<Geometry*> *geoms);

void combineLayers(polygonLayerTmp* send, polygonLayerTmp* recv){
	recv->polNum += send->polNum;

	for(vector<int >::iterator itr = send->vNum->begin(); itr != send->vNum->end(); ++itr){
		recv->vNum->push_back(*itr);
	}

    int tmpPrefix = 0;
    if(! recv->prefixSum->empty()){
		tmpPrefix = recv->prefixSum->back();
	}
	
	for(vector<long >::iterator itr = send->prefixSum->begin(); itr != send->prefixSum->end(); ++itr){
		if(!((tmpPrefix!=0) && ((*itr)==0)))
		recv->prefixSum->push_back(*itr+tmpPrefix);
	}
	
	for(vector<Coordinate >::iterator itr = send->coord->begin(); itr != send->coord->end(); ++itr){
		recv->coord->push_back(*itr);
	}
	
	for(vector<Coordinate>::iterator itr = send->mbr->begin(); itr != send->mbr->end(); ++itr){
		recv->mbr->push_back(*itr);
	}

	for(vector<Envelope>::iterator itr = send->mbr2->begin(); itr != send->mbr2->end(); ++itr){
		recv->mbr2->push_back(*itr);
	}
}

void parsing(vector<string> *vstr1, int start1, int end1, vector<string> *vstr2, int start2, int end2, polygonLayerTmp* layer1, polygonLayerTmp* layer2){

    vector<string> *localStr1 = new vector<string>(vstr1->begin()+start1, vstr1->begin()+end1-1);

    vector<string> *localStr2 = new vector<string>(vstr2->begin()+start2, vstr2->begin()+end2-1);

    list<Geometry*>* lGeos1 = new list<Geometry*>;
    list<Geometry*>* lGeos2 = new list<Geometry*>;

    readGeomsFromStr(localStr1, lGeos1);
    readGeomsFromStr(localStr2, lGeos2);

    polygonLayerTmp* tmpLayer1 = populateLayerData(lGeos1);
	polygonLayerTmp* tmpLayer2 = populateLayerData(lGeos2);
	
	push_mutex.lock();
	combineLayers(tmpLayer1,layer1);
	combineLayers(tmpLayer2,layer2);
	push_mutex.unlock();
}

void convertMBRToFloats(const Envelope* v, vector<Coordinate> *vertVect)
{	
	Coordinate P1,P2;
	P1.x =(double)v->getMinX();
	P1.y =(double)v->getMinY();
	P2.x =(double)v->getMaxX();
	P2.y =(double)v->getMaxY();
    vertVect->push_back(P1);
    vertVect->push_back(P2);
}		

void convertToFloats(const LineString* vertices, vector<Coordinate> *vertVect)
{
	size_t numPoints = vertices->getNumPoints();
  
	for(size_t i = 0; i<numPoints; i++) 
	{
		Point* pt = vertices->getPointN(i);
		Coordinate P3;
		P3.x = pt->getX();
		P3.y = pt->getY();
		vertVect->push_back(P3);
	} 
}

vector<long> * prefixSum(polygonLayerTmp *layer)
{
	long numPoly = layer->polNum;
	vector<int>* vNum = layer->vNum;
   	vector<long>* prefixsum = new vector<long>;
	prefixsum->push_back(0);
   
	for(long i = 1; i < numPoly+1; i++) 
	{
		prefixsum->push_back(prefixsum->at(i-1)+vNum->at(i-1));
	} 
	return prefixsum;
}

void gpuHelperForMultiPolygon(Geometry *geom, vector<Coordinate> *verticesVec, 
		vector<Coordinate> *envVec, vector<Envelope> *gpuEnvInLongVector, vector<int> *vNumVector)
{
	size_t numGeoms = geom->getNumGeometries();
         
	for(size_t i = 0; i < numGeoms; i++) 
	{
		const Geometry* inner = geom->getGeometryN(i);
		
        const Polygon* poly = dynamic_cast<const Polygon*>(inner);      
		const LineString *innerLinestring = poly->getExteriorRing();
        
		vNumVector->push_back(innerLinestring->getNumPoints());		
		convertToFloats(innerLinestring, verticesVec);   	
		convertMBRToFloats(poly->getEnvelopeInternal(), envVec);		
	}
}

void gpuHelperForPolygon(Geometry *geom, vector<Coordinate> *verticesVec, 
		vector<Coordinate> *envVec, vector<Envelope> *gpuEnvInLongVector, vector<int> *vNumVector)
{
	Polygon* poly = dynamic_cast<Polygon*>(geom); 
	const LineString *linestring = poly->getExteriorRing();
   
	vNumVector->push_back(linestring->getNumPoints());
	convertToFloats(linestring, verticesVec);     
	convertMBRToFloats(poly->getEnvelopeInternal(), envVec);		
}

polygonLayerTmp* populateLayerData(list<Geometry*>* geoms)
{	
	polygonLayerTmp* layer = (polygonLayerTmp*)malloc(1* sizeof(polygonLayerTmp)); 
   
	vector<Coordinate> *runningVector = new vector<Coordinate>();	
	vector<Coordinate> *runningEnvVector = new vector<Coordinate>();  
	vector<Envelope> *gpuEnvInLongVector = new vector<Envelope>();
	vector<int> *vNumVector = new vector<int>();  

	int numPolygons = 0;  
      
	for(list<Geometry*>::iterator it = geoms->begin() ; it != geoms->end(); ++it) 
	{
		Geometry *geom = *it;
		GeometryTypeId typeId = geom->getGeometryTypeId();
    
		switch(typeId)
		{
            case GEOS_POLYGON:
			{
				gpuHelperForPolygon(geom, runningVector, runningEnvVector, gpuEnvInLongVector, vNumVector);
				numPolygons++;
			} 
			break;
       
			case GEOS_MULTIPOLYGON:
			{
				gpuHelperForMultiPolygon(geom, runningVector, runningEnvVector, gpuEnvInLongVector, vNumVector);
			}
			break;
       
			case GEOS_GEOMETRYCOLLECTION:
			{
				size_t numGeoms = geom->getNumGeometries();
				numPolygons += numGeoms;       
       
				for(size_t i = 0; i < numGeoms; i++)
				{
					GeometryTypeId typeId = geom->getGeometryTypeId();
         
					switch(typeId)
					{
						case GEOS_POLYGON:
						{
							gpuHelperForPolygon(geom, runningVector, runningEnvVector, gpuEnvInLongVector, 
								vNumVector);
							numPolygons++;
						} 
						break;
       
						case GEOS_MULTIPOLYGON:
						{
							gpuHelperForMultiPolygon(geom, runningVector, runningEnvVector, gpuEnvInLongVector, 
								vNumVector);
						}
						break;
					} 
				} 
			}
			break;  
		}
	}
	layer->polNum = vNumVector->size();     
	layer->vNum = vNumVector;     
	layer->prefixSum = prefixSum(layer);	
	layer->coord = runningVector;   
	layer->mbr = runningEnvVector;  
	layer->mbr2 = gpuEnvInLongVector;
	return layer;
}

void readGeomsFromStr(vector<string> *vstr, list<Geometry*> *geoms){
    geos::io::WKTReader wktreader;

    for(vector<string >::iterator itr = vstr->begin(); itr != vstr->end(); ++itr){
        string tmpStr = *itr;
        try{
		    Geometry* tmpGeo = NULL;
		    tmpGeo = wktreader.read(tmpStr);
		    if(tmpGeo != NULL && tmpGeo->isValid())
			    geoms->push_back(tmpGeo);
		}catch(exception &e){
		}
    }
    
}
//above functions get the basic information for one data set 

int MySearchCallback2(int id, void* arg) 
 {
       int* idOfbase = (int *)arg;
	
       if(subjectPoly2[*idOfbase].num_elementsBasePoly == subjectPoly2[*idOfbase].num_allocatedBasePoly)
       {
			if (subjectPoly2[*idOfbase].num_allocatedBasePoly == 0)
            subjectPoly2[*idOfbase].num_allocatedBasePoly = 100; 
			else
			{
				subjectPoly2[*idOfbase].num_allocatedBasePoly += 100; 
			}
			void *_tmp = realloc(subjectPoly2[*idOfbase].overlayPolyIndices, (subjectPoly2[*idOfbase].num_allocatedBasePoly 
			* sizeof(int)));
			if (!_tmp)
			{
				printf("************ ERROR: Couldn't realloc memory!***********\n");
				fflush(stdout);
			}
			subjectPoly2[*idOfbase].overlayPolyIndices = (int *)_tmp;   
       }
	   subjectPoly2[*idOfbase].overlayPolyIndices[subjectPoly2[*idOfbase].num_elementsBasePoly] = id-1;
       subjectPoly2[*idOfbase].num_elementsBasePoly++;
       return 1;
}

void rtreeBuildingAndSearch2(int num_contours, double *rect1, double *rect2, double *rect3,double *rect4, 
 int numOfQuerys,double *rect1_query,double *rect2_query, double *rect3_query, double *rect4_query)
{
   subjectPoly2 = (gpc_overlayPoly2 *)malloc(num_contours * 
			  sizeof(gpc_overlayPoly2));
  
    struct Node* root = RTreeNewIndex();
    int nhits,i;
			  
	for(i=0; i<numOfQuerys; i++)
	{
	    struct Rect rect = {rect1_query[i], rect2_query[i], rect3_query[i], rect4_query[i]};
        RTreeInsertRect(&rect, i+1, &root, 0);
    }
	int c = 0;
	
	for (c= 0; c < num_contours; c++)
    {
      gpc_overlayPoly2 p;
      p.num_elementsBasePoly = 0;
      p.num_allocatedBasePoly = 0;
      p.overlayPolyIndices = NULL;
      subjectPoly2[c] = p;
	}
					
 	int a;
	
	for(a = 0; a <num_contours; a++)
    {
	  nhits = 0;
	  struct Rect search_rect = {rect1[a], rect2[a], rect3[a], rect4[a]};
	  nhits = RTreeSearch(root, &search_rect, MySearchCallback2, &a);
	}
}

int localProcessing48(int num_contours, double *rect1, double *rect2, double *rect3,double *rect4, int *id_base,
		int numOfQuerys,double *rect1_query,double *rect2_query, double *rect3_query,
		double *rect4_query,int *id_query,int *num_base,int *prefix_base,gpc_vertex *vertex_base,
		int *num_overlay,int *prefix_overlay,gpc_vertex *vertex_overlay,int *hole_base_cpu,
		int *hole_overlay_cpu,int m,int M,int nprocs,int minX,int minY,char * name)
{	
	double starttime0, endtime0;
    double difference0;	
    starttime0 = my_difftime();
	
    rtreeBuildingAndSearch(num_contours, rect1, rect2, rect3, rect4,
     id_base, numOfQuerys, rect1_query, rect2_query, rect3_query, rect4_query);

	int tasks = 0; 
	int clipIndex;
    int a;
             
	for(a = 0; a < num_contours; a++)
	{
        tasks = tasks + subjectPoly[a].num_elementsBasePoly;
    }
       
    int *htaskSubId = (int *)malloc(tasks * sizeof(int));
    int *htaskClipId = (int *)malloc(tasks * sizeof(int)); 
    int counter = 0;
	
    for(a = 0; a < num_contours; a++)
	{
        for(clipIndex = 0; clipIndex < subjectPoly[a].num_elementsBasePoly; clipIndex++)
        {
          htaskSubId[counter] = a;
          htaskClipId[counter] = subjectPoly[a].overlayPolyIndices[clipIndex];
          counter = counter + 1;
        }
    }
	endtime0 = my_difftime();
	difference0 = endtime0 - starttime0;
	printf("Rtree time taken =  %f\t \n",difference0);
	printf("total candidate tasks = %d \n", tasks);
	
	//transfer from double to long long 
	gpc_vertex2* ha1 = transfer_pointsvalues_cuda3(num_contours, num_base, prefix_base, vertex_base);
	gpc_vertex2* hb1 = transfer_pointsvalues_cuda3(numOfQuerys, num_overlay, prefix_overlay, vertex_overlay);
	
	long long* rect1n = transfer_boundingbox_cuda3(num_contours, rect1);
	long long* rect2n = transfer_boundingbox_cuda3(num_contours, rect2);
	long long* rect3n = transfer_boundingbox_cuda3(num_contours, rect3);
	long long* rect4n = transfer_boundingbox_cuda3(num_contours, rect4);
	long long* rect1_queryn = transfer_boundingbox_cuda3(numOfQuerys, rect1_query);
	long long* rect2_queryn = transfer_boundingbox_cuda3(numOfQuerys, rect2_query);
	long long* rect3_queryn = transfer_boundingbox_cuda3(numOfQuerys, rect3_query);
	long long* rect4_queryn = transfer_boundingbox_cuda3(numOfQuerys, rect4_query);
		
	//for department's gpu to wake up
	int numWake = 4096000;
	wakeGPUup1(numWake);
	
	//double starttime2, endtime2;
    //double difference2;	
    //starttime2 = my_difftime();
			
	//PolySketch-CMBR
	//int *SIresult = pSketch23(tasks, htaskSubId, htaskClipId, num_contours, numOfQuerys, num_base, num_overlay, 
	//		prefix_base, prefix_overlay, ha1, hb1,rect1n,rect2n,rect3n,rect4n,rect1_queryn,rect2_queryn,rect3_queryn,
	//		rect4_queryn);	
	
	//PolySketch-CMBR with PNP test
	int *SIresult = pSketch26(tasks, htaskSubId, htaskClipId, num_contours, numOfQuerys, num_base, num_overlay, 
			prefix_base, prefix_overlay, ha1, hb1,rect1n,rect2n,rect3n,rect4n,rect1_queryn,rect2_queryn,rect3_queryn,
			rect4_queryn);	
									
	//endtime2 = my_difftime();
	//difference2 = endtime2 - starttime2;
	//printf("CUDA time taken =  %f\t \n",difference2);
			
	return 0;
}

//for pscmbr wkt data
int localProcessing60(int num_contours, double *rect1, double *rect2, double *rect3,double *rect4, 
		int numOfQuerys,double *rect1_query,double *rect2_query, double *rect3_query,
		double *rect4_query,int *num_base,long *prefix_base,Coordinate *vertex_base,
		int *num_overlay,long *prefix_overlay,Coordinate *vertex_overlay)
{
	double starttime0, endtime0;
    double difference0;	
    starttime0 = my_difftime();
	
    rtreeBuildingAndSearch2(num_contours, rect1, rect2, rect3, rect4,
     numOfQuerys, rect1_query, rect2_query, rect3_query, rect4_query);

	int tasks = 0; 
    int clipIndex;
    int a;
             
	for(a = 0; a < num_contours; a++)
	{
        tasks = tasks + subjectPoly2[a].num_elementsBasePoly;
    }
    
    int *htaskSubId = (int *)malloc(tasks * sizeof(int));
    int *htaskClipId = (int *)malloc(tasks * sizeof(int)); 
    int counter = 0;
	
    for(a = 0; a < num_contours; a++)
	{
        for(clipIndex = 0; clipIndex < subjectPoly2[a].num_elementsBasePoly; clipIndex++)
        {
          htaskSubId[counter] = a;
          htaskClipId[counter] = subjectPoly2[a].overlayPolyIndices[clipIndex];
          counter = counter + 1;
        }
    }
	endtime0 = my_difftime();
	difference0 = endtime0 - starttime0;
	printf("Rtree time taken =  %f\t \n",difference0);
	printf("total tasks = %d \n", tasks);
	
	//transfer from double to long long 
	gpc_vertex2* ha1 = transfer_pointsvalues_cuda4(num_contours, num_base, prefix_base, vertex_base);	
	gpc_vertex2* hb1 = transfer_pointsvalues_cuda4(numOfQuerys, num_overlay, prefix_overlay, vertex_overlay);
	
	long long* rect1n = transfer_boundingbox_cuda3(num_contours, rect1);
	long long* rect2n = transfer_boundingbox_cuda3(num_contours, rect2);
	long long* rect3n = transfer_boundingbox_cuda3(num_contours, rect3);
	long long* rect4n = transfer_boundingbox_cuda3(num_contours, rect4);
	long long* rect1_queryn = transfer_boundingbox_cuda3(numOfQuerys, rect1_query);
	long long* rect2_queryn = transfer_boundingbox_cuda3(numOfQuerys, rect2_query);
	long long* rect3_queryn = transfer_boundingbox_cuda3(numOfQuerys, rect3_query);
	long long* rect4_queryn = transfer_boundingbox_cuda3(numOfQuerys, rect4_query);
	
	int numWake = 4096000;
	wakeGPUup1(numWake);
	
	numWake = 40960;
	wakeGPUup1(numWake);
	
	//PolySketch-CMBR for wkt dataset
	int *SIresult = pSketch25(tasks, htaskSubId, htaskClipId, num_contours, numOfQuerys, num_base, num_overlay, 
									prefix_base, prefix_overlay, ha1, hb1,rect1n,rect2n,rect3n,rect4n,rect1_queryn,rect2_queryn,rect3_queryn,
									rect4_queryn);	
	
	
	return 0;
}

int readShapefile(int argc, char *argv[])
{    
	int i,j;

	SHPHandle	hSHP_base;
	hSHP_base = SHPOpen(argv[1],"rb");
	PolyRect *mbrs_base;
	
	int startIndex_base=0;
    int endIndex_base=hSHP_base->nRecords-1;
    int count_base=endIndex_base-startIndex_base+1;   
	SHPReadMBR1(hSHP_base, startIndex_base, endIndex_base, &mbrs_base);
	
	SHPHandle	hSHP_overlay;
	hSHP_overlay = SHPOpen(argv[2],"rb");
	PolyRect *mbrs_overlay;
	
	int startIndex_overlay=0;
    int endIndex_overlay=hSHP_overlay->nRecords-1;
    int count_overlay=endIndex_overlay-startIndex_overlay+1;
	SHPReadMBR1(hSHP_overlay, startIndex_overlay, endIndex_overlay, &mbrs_overlay);
	
	bBox *baseBoxes;
	bBox *overlayBoxes;
	
	baseBoxes=(bBox *)malloc(sizeof(bBox));
	overlayBoxes=(bBox *)malloc(sizeof(bBox));
	baseBoxes->count=count_base;
	baseBoxes->rects=mbrs_base;
	overlayBoxes->count=count_overlay;
	overlayBoxes->rects=mbrs_overlay;	
	
	int sum_mbrs_base=baseBoxes[0].count ;
	int sum_mbrs_overlay=overlayBoxes[0].count;
	
	//Base layer 
	double minX_base,minY_base;
	minX_base=10000000;
	minY_base=10000000;
	SHPObject	**psShape_base;
	psShape_base=(SHPObject **)malloc((sum_mbrs_base)*sizeof(SHPObject*));
	
	int *num_base;
	int * prefix_base;
	int * hole_base_cpu=(int *)malloc((sum_mbrs_base)*sizeof(int));
	for(i=0;i<sum_mbrs_base;i++){
		hole_base_cpu[i]=0;
	}
	num_base=(int *)malloc((sum_mbrs_base)*sizeof(int));
	prefix_base=(int *)malloc((sum_mbrs_base)*sizeof(int));
	
	double 	*rect1=(double *)malloc(sum_mbrs_base*sizeof(double));
 	double	*rect2=(double *)malloc(sum_mbrs_base*sizeof(double));
 	double	*rect3=(double *)malloc(sum_mbrs_base*sizeof(double));
 	double	*rect4=(double *)malloc(sum_mbrs_base*sizeof(double));	
	convert(rect1,rect2,rect3,rect4,psShape_base,num_base,prefix_base,baseBoxes,0,
	sum_mbrs_base,hSHP_base,&minX_base,&minY_base);
	gpc_vertex *vertex_base_cpu=(gpc_vertex *)malloc((prefix_base[sum_mbrs_base-1]+num_base[sum_mbrs_base-1])
	*sizeof(gpc_vertex));
	for(i =0 ; i< sum_mbrs_base;i++)
	{
		for(j=0;j<num_base[i];j++)
		{
			vertex_base_cpu[prefix_base[i]+j].x=psShape_base[i]->padfX[j];
			vertex_base_cpu[prefix_base[i]+j].y=psShape_base[i]->padfY[j];
		}
	}
	
	destoryObjects(psShape_base,sum_mbrs_base);

	double minX_overlay,minY_overlay;
	minX_overlay=10000000;
	minY_overlay=10000000;   
	SHPObject	**psShape_overlay;
	psShape_overlay=(SHPObject **)malloc((sum_mbrs_overlay)*sizeof(SHPObject*));
	
	int *num_overlay;
	int * prefix_overlay;
	int * hole_overlay_cpu=(int *)malloc((sum_mbrs_overlay)*sizeof(int));
	for(i=0;i<overlayBoxes[0].count;i++){
		hole_overlay_cpu[i]=0;
	}
	num_overlay=(int *)malloc((sum_mbrs_overlay)*sizeof(int));
	prefix_overlay=(int *)malloc((sum_mbrs_overlay)*sizeof(int));
	
	double * rect1_query_cpu;
	double * rect2_query_cpu;
	double * rect3_query_cpu;
	double * rect4_query_cpu;
	rect1_query_cpu=(double *)malloc(sum_mbrs_overlay*sizeof(double));
 	rect2_query_cpu=(double *)malloc(sum_mbrs_overlay*sizeof(double));
 	rect3_query_cpu=(double *)malloc(sum_mbrs_overlay*sizeof(double));
 	rect4_query_cpu=(double *)malloc(sum_mbrs_overlay*sizeof(double));	
	convert(rect1_query_cpu,rect2_query_cpu,rect3_query_cpu,rect4_query_cpu,
        psShape_overlay,num_overlay,prefix_overlay,overlayBoxes,0,sum_mbrs_overlay,
        hSHP_overlay,&minX_overlay,&minY_overlay);
	gpc_vertex *vertex_overlay_cpu=(gpc_vertex *)malloc((prefix_overlay[sum_mbrs_overlay-1]+
	num_overlay[sum_mbrs_overlay-1])*sizeof(gpc_vertex));
	for(i =0 ; i< sum_mbrs_overlay;i++)
	{
		for(j=0;j<num_overlay[i];j++)
		{   
			vertex_overlay_cpu[prefix_overlay[i]+j].x=psShape_overlay[i]->padfX[j];
			vertex_overlay_cpu[prefix_overlay[i]+j].y=psShape_overlay[i]->padfY[j];
		}
	}
	
	destoryObjects(psShape_overlay,sum_mbrs_overlay);
			
	localProcessing48(sum_mbrs_base, rect1,rect2,rect3,rect4,NULL,sum_mbrs_overlay,
		rect1_query_cpu,rect2_query_cpu,rect3_query_cpu,rect4_query_cpu,NULL,num_base,
		prefix_base,vertex_base_cpu,num_overlay,prefix_overlay,vertex_overlay_cpu,
		hole_base_cpu,hole_overlay_cpu,6,12,0,minX_base,minY_base,NULL);

	SHPClose(hSHP_base);	
	SHPClose(hSHP_overlay);	
	
	return 0;
} 

//for wkt data
int readNewFile(string filePath1, string filePath2){
	
	double starttime0001, endtime0001;
    double difference0001;	
    starttime0001 = my_difftime();
	
	vector<string> *vstr1 = new vector<string>;
    vector<string> *vstr2 = new vector<string>;
    string tmpStr;

    ifstream file1(filePath1.c_str());
    while(std::getline(file1, tmpStr)){
        vstr1->push_back(tmpStr);
    }
    file1.close();

    ifstream file2(filePath2.c_str());
    while(std::getline(file2, tmpStr)){
        vstr2->push_back(tmpStr);
    }
    file2.close();

	polygonLayerTmp* tmpLayer1 = (polygonLayerTmp*)malloc(1* sizeof(polygonLayerTmp));
	tmpLayer1->polNum = 0;
    tmpLayer1->vNum = new vector<int>;
    tmpLayer1->prefixSum = new vector<long>;
    tmpLayer1->coord = new vector<Coordinate>;
    tmpLayer1->mbr = new vector<Coordinate>;
    tmpLayer1->mbr2 = new vector<Envelope>;
    
    polygonLayerTmp* tmpLayer2 = (polygonLayerTmp*)malloc(1* sizeof(polygonLayerTmp));
	tmpLayer2->polNum = 0;
    tmpLayer2->vNum = new vector<int>;
    tmpLayer2->prefixSum = new vector<long>;
    tmpLayer2->coord = new vector<Coordinate>;
    tmpLayer2->mbr = new vector<Coordinate>;
    tmpLayer2->mbr2 = new vector<Envelope>;
   
    int numThreads = 36;
    thread * parseThread = new thread[numThreads];

    int numOfStrPerThread1 = vstr1->size() / numThreads;
	int numOfStrPerThread2 = vstr2->size() / numThreads;
    
    for (int i = 0; i < numThreads; i++){
        int start1 = i * numOfStrPerThread1;
        int end1 = (i+1) * numOfStrPerThread1;
        int start2 = i * numOfStrPerThread2;
        int end2 = (i+1) * numOfStrPerThread2;
        if(numThreads - 1 == i){ 
            end1 = vstr1->size();
            end2 = vstr2->size();
        }
        parseThread[i] = thread(parsing, vstr1, start1, end1, vstr2, start2, end2, tmpLayer1, tmpLayer2);
    }

    for (int i = 0; i < numThreads; i++){
         parseThread[i].join();
     }
     delete [] parseThread;

	int polNum1 = tmpLayer1->polNum;
	int* vNum1 = tmpLayer1->vNum->data();
	long* prefixSum1 = tmpLayer1->prefixSum->data();		
	Coordinate* coord1 = tmpLayer1->coord->data();		
	vector<Coordinate>* mbr21 = tmpLayer1->mbr;		
	
	int polNum2 = tmpLayer2->polNum;
	int* vNum2 = tmpLayer2->vNum->data();
	long* prefixSum2 = tmpLayer2->prefixSum->data();		
	Coordinate* coord2 = tmpLayer2->coord->data();		
	vector<Coordinate> *mbr22 = tmpLayer2->mbr;			

	double *rect1 = (double*)malloc(polNum1*sizeof(double));
    double *rect2 = (double*)malloc(polNum1*sizeof(double));
    double *rect3 = (double*)malloc(polNum1*sizeof(double));
    double *rect4 = (double*)malloc(polNum1*sizeof(double));
    double *rect1_query = (double*)malloc(polNum2*sizeof(double));
    double *rect2_query = (double*)malloc(polNum2*sizeof(double));
    double *rect3_query = (double*)malloc(polNum2*sizeof(double));
    double *rect4_query = (double*)malloc(polNum2*sizeof(double));

	for(int idx=0;idx<(polNum1);idx++){
		rect1[idx]= mbr21->at(idx*2).x;
		rect2[idx]= mbr21->at(idx*2).y;
		rect3[idx]= mbr21->at(idx*2+1).x;
		rect4[idx]= mbr21->at(idx*2+1).y;
	}
	for(int idx=0;idx<(polNum2);idx++){
		rect1_query[idx]= mbr22->at(idx*2).x;
		rect2_query[idx]= mbr22->at(idx*2).y;
		rect3_query[idx]= mbr22->at(idx*2+1).x;
		rect4_query[idx]= mbr22->at(idx*2+1).y;
	}

	endtime0001 = my_difftime();
	difference0001 = endtime0001 - starttime0001;
	printf("Reading data time taken =  %f\t \n",difference0001);
	
	//pscmbr for wkt
	localProcessing60(polNum1, rect1,rect2,rect3,rect4,polNum2,
		rect1_query,rect2_query,rect3_query,rect4_query,vNum1,
		prefixSum1,coord1,vNum2,prefixSum2,coord2);

	printf("  \n");
	
	return 0;
}

int main(int argc, char *argv[])
{ 		
	//readShapefile(argc, argv);
	
	//for wkt data  
	string A1 = "/home/yliu0204/lakes_data";
	string A2 = "/home/yliu0204/sports_data";
	
	readNewFile(A1,A2);  
	
	return 0;
}
