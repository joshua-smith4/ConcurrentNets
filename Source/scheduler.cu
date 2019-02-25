/*
 * scheduler.cu
 *
 *  Created on: Sep 22, 2011
 *      Author: yiding
 */
#include <iostream>
#include <algorithm>
#include <cstdlib>

using std::cout;
using std::endl;
using std::flush;
using std::min;

#include "GPUKernels.cu"
#include "scheduler.h"

const uint NUM_BLOCKS = 10;
const uint THREADS_PER_BLOCK = 10;

Scheduler::Scheduler(DB& _db, const CommandLine& _params) :  db(_db), params(_params)
{
	//initialization
	hostTiles = (IdType *) malloc(db.xTiles * db.yTiles * sizeof(IdType));

}

/* driver code starts here */
int Scheduler::run()
{
	//Find GPU hardware
	vector<int> CUDADeviceList;
	const unsigned cudaMajorVer = CUDA_MAJOR_VER;
	const unsigned cudaMinorVer = CUDA_MINOR_VER;
	if (params.allowGPU) {
		findCUDADevice(CUDADeviceList, cudaMajorVer, cudaMinorVer);
		if (CUDADeviceList.size() == 0) {
			cout << "Error: No CUDA " << cudaMajorVer << "." << cudaMinorVer << " or later device available." << endl;
			exit(0);
		}
	}

	WallClockTimer timer;	//Timer variable
	SubNetQueue queue;		//we will save a queue of subnets in this container

	timer.start();

	//Create a container includes all subnets' ID
	for (unsigned i = 0; i < db.nets.size(); ++i) {
		for (unsigned j = 0; j < db.nets[i].subNets.size(); ++j) {
			queue.push_back(make_pair(i, j));
		}
	}
	//Sort all subnets with their size and aspect ratio
	sort(queue.begin(), queue.end(), CompareByBox(db.nets));

	ProgressIndicator progress(queue.size(), 0, params.printProgress);	//this object posts progress to user

	//this loop is the core of this program: find out concurrent subnets from a ordered subnet queue
	while(queue.size() != 0) {
		db.concurrentSubnets.push_back(SubNetQueue());
		if (params.allowGPU)
			findConcurrencyGPU(queue, db.concurrentSubnets.back(), params.schedulingWindow);
		else
			findConcurrencyCPU(queue, db.concurrentSubnets.back(), params.schedulingWindow);
		progress.print(queue.size(), "Extracting Concurrent Nets");
	}

	timer.stop();

	cout << endl << "Total computational time: " << timer.reading() << " s" << endl << endl;
	return 0;
}

/*
 * Look for CUDA device with specific compute capability
 * (1.0, 1.1, 1.2, 1.3, 2.0)
 * returns the number of CUDA capable devices. The device number that
 * satisfies the required compute capability are saved to deviceList.
 */
int Scheduler::findCUDADevice(vector<int>& deviceList,
		const unsigned minMajorVer,
		const unsigned minMinerVer)
{
	// check the compute capability of the device
	int numDevices = 0;
	cudaError_t retval = cudaGetDeviceCount(&numDevices);

	if (retval == cudaErrorNoDevice) {
		cout << "No CUDA capable device found in your system." << endl;
		deviceList.clear();
		return 0;
	}

	if (retval == cudaErrorInsufficientDriver) {
		cout << "No driver can be loaded to determine if any CUDA devices exist." << endl;
		deviceList.clear();
		return 0;
	}

	if (retval != cudaSuccess) {
		cout << "Previous asynchronous launch error exists." << endl;
		deviceList.clear();
		return 0;
	}

	for (int device = 0; device < numDevices; device++) {
		cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, device);

		cout << "Found CUDA " << properties.major << "." << properties.minor << " device" << endl;

		if ((properties.major <= minMajorVer) && (properties.minor < minMinerVer)) {
		}
		else {
			deviceList.push_back(device);
		}
	}

	return numDevices;
}

Scheduler::~Scheduler(void)
{
	// free resources
	free(hostTiles);

}

int Scheduler::findConcurrencyCPU(SubNetQueue& subNetsQueue, SubNetQueue& concurrentSubNets, const size_t windowSize)
{
	size_t subNetCount = min(windowSize, subNetsQueue.size());
	if (subNetCount == 0) {
		concurrentSubNets.clear();
		return 0;
	}

	//hint: uint2 is a CUDA data type, it is supported on both CPU and GPU
	uint2 a, b;
	Point A, B;
	vector<uint2> hostA, hostB;
	hostA.reserve(subNetCount);
	hostB.reserve(subNetCount);
	unsigned minX = db.xTiles;
	unsigned maxX = 0;
	unsigned minY = db.yTiles;
	unsigned maxY = 0;

	//Preparation
	int i = 0;
	for (SubNetQueue::reverse_iterator it = subNetsQueue.rbegin(); i < subNetCount; it++, i++) {
		SubNet& subnet = db.nets[(*it).first].subNets[(*it).second];
		A = subnet.a;
		B = subnet.b;
		a.x = min(A.x, B.x);
		a.y = min(A.y, B.y);
		b.x = max(A.x, B.x);
		b.y = max(A.y, B.y);

		hostA.push_back(a);
		hostB.push_back(b);

		if (minX > a.x)
			minX = a.x;
		if (minY > a.y)
			minY = a.y;
		if (maxX < b.x)
			maxX = b.x;
		if (maxY < b.y)
			maxY = b.y;
	}
	std::cout << maxY - minY << "\n";
	std::cout << maxX - minX << "\n";
	//color the Tiles
	vector< vector<IdType> > colorTiles;
	colorTiles.resize(db.yTiles);
	for (vector< vector<IdType> >::iterator it = colorTiles.begin(); it != colorTiles.end(); it++) {
		it->resize(db.xTiles, NO_ID);
	}
	for (int y = minY; y <= maxY; y++) {
		for (int x = minX; x <= maxX; x++) {
			for (int i = 0; i < subNetCount; i++) {
				if (hostA[i].x <= x && hostB[i].x >= x && hostA[i].y <= y && hostB[i].y >= y) {
					colorTiles[y][x] = i;
					break;
				}
			}
		}
	}

	//count all subnets marked on the colored Tile
	vector<unsigned> tilesWithinRoutingRegion(subNetCount, 0);
	for (int y = minY; y <= maxY; y++) {
			for (int x = minX; x <= maxX; x++) {
			if (colorTiles[y][x] != NO_ID) {
				tilesWithinRoutingRegion[colorTiles[y][x]]++;
			}
		}
	}

	//find out the concurrent subnets. Push them into concurrentSubNet, and erase from the subNetsQueue
	concurrentSubNets.clear();
	SubNetQueue::reverse_iterator it = subNetsQueue.rbegin();
	for (i = 0; i < subNetCount; i++) {
		if (tilesWithinRoutingRegion[i] != 0
				&& tilesWithinRoutingRegion[i] == (hostB[i].x - hostA[i].x + 1) * (hostB[i].y - hostA[i].y + 1)) {
			concurrentSubNets.push_back(*it);
			subNetsQueue.erase((++it).base());
		}
		else {
			it++;
		}
	}

	return concurrentSubNets.size();
}



int Scheduler::findConcurrencyGPU(SubNetQueue& subNetsQueue, SubNetQueue& concurrentSubNets, const size_t windowSize)
{
	//GPU implementation of findConcurrencyCPU()
	// size_t subNetCount = min(windowSize, subNetsQueue.size());
	// if (subNetCount == 0) {
	// 	concurrentSubNets.clear();
	// 	return 0;
	// }
	//
	// uint2 a, b;
	// Point A, B;
	// auto hostA = new uint2[subNetCount];
	// auto hostB = new uint2[subNetCount];
	// // vector<uint2> hostA, hostB;
	// // hostA.reserve(subNetCount);
	// // hostB.reserve(subNetCount);
	// unsigned minX = db.xTiles;
	// unsigned maxX = 0;
	// unsigned minY = db.yTiles;
	// unsigned maxY = 0;
	//
	// //Preparation
	// int i = 0;
	// for (SubNetQueue::reverse_iterator it = subNetsQueue.rbegin(); i < subNetCount; it++, i++) {
	// 	SubNet& subnet = db.nets[(*it).first].subNets[(*it).second];
	// 	A = subnet.a;
	// 	B = subnet.b;
	// 	a.x = min(A.x, B.x);
	// 	a.y = min(A.y, B.y);
	// 	b.x = max(A.x, B.x);
	// 	b.y = max(A.y, B.y);
	//
	// 	hostA[i] = a;
	// 	hostB[i] = b;
	//
	// 	if (minX > a.x)
	// 		minX = a.x;
	// 	if (minY > a.y)
	// 		minY = a.y;
	// 	if (maxX < b.x)
	// 		maxX = b.x;
	// 	if (maxY < b.y)
	// 		maxY = b.y;
	// }
	//
	// auto colorTiles = new IdType*[db.yTiles];
	// IdType** deviceColorTiles;
	// cudaMalloc(&deviceColorTiles, sizeof(IdType*)*db.yTiles);
	// for(unsigned j = 0; j < db.yTiles; ++j)
	// {
	// 	colorTiles[j] = new IdType[db.xTiles];
	// 	cudaMalloc(&deviceColorTiles[j], sizeof(IdType)*db.xTiles);
	// 	for(unsigned k = 0; k < db.xTiles; ++k)
	// 	{
	// 		colorTiles[j][k] = NO_ID;
	// 	}
	// 	cudaMemcpy(deviceColorTiles[j], colorTiles[j], sizeof(IdType)*db.xTiles, cudaMemcpyHostToDevice);
	// }
	//
	// std::size_t size = sizeof(uint2)*subNetCount;
	//
	// uint2* deviceA;
	// cudaMalloc(&deviceA, sizeof(uint2)*subNetCount);
	// uint2* deviceB;
	// cudaMalloc(&deviceB, sizeof(uint2)*subNetCount);
	//
	// cudaMemcpy(deviceA, hostA, size, cudaMemcpyHostToDevice);
	// cudaMemcpy(deviceB, hostB, size, cudaMemcpyHostToDevice);
	//
	// auto hostTilesWithinRoutingRegion = new unsigned[subNetCount];
	// unsigned* deviceTilesWithinRoutingRegion;
	// cudaMalloc(&deviceTilesWithinRoutingRegion, sizeof(unsigned)*subNetCount);
	// cudaMemset(deviceTilesWithinRoutingRegion, 0, sizeof(unsigned)*subNetCount);
	//
	//
	// // gather return values
	// cudaMemcpy(hostTilesWithinRoutingRegion, deviceTilesWithinRoutingRegion, sizeof(unsigned)*subNetCount, cudaMemcpyDeviceToHost);
	//
	// // deallocate memory when no longer in use
	// for(unsigned j = 0; j < db.yTiles; ++j)
	// {
	// 	delete[] colorTiles[j];
	// }
	// delete[] colorTiles;
	//
	// delete[] hostA;
	// delete[] hostB;
	// // std::free(tilesWithinRoutingRegion);
	//
	// cudaFree(deviceA);
	// cudaFree(deviceB);
	//
	// for(unsigned j = 0; j < db.yTiles; ++j)
	// {
	// 	cudaFree(deviceColorTiles[j]);
	// }
	// cudaFree(deviceColorTiles);
	//
	// concurrentSubNets.clear();
	// SubNetQueue::reverse_iterator it = subNetsQueue.rbegin();
	// for (i = 0; i < subNetCount; i++) {
	// 	if (hostTilesWithinRoutingRegion[i] != 0
	// 			&& hostTilesWithinRoutingRegion[i] == (hostB[i].x - hostA[i].x + 1) * (hostB[i].y - hostA[i].y + 1)) {
	// 		concurrentSubNets.push_back(*it);
	// 		subNetsQueue.erase((++it).base());
	// 	}
	// 	else {
	// 		it++;
	// 	}
	// }

	return concurrentSubNets.size();
}

bool CompareByBox::operator()(const SubNetIdType &a, const SubNetIdType &b) const
{
	CostType aW = fabs(static_cast<CostType> (nets[a.first].subNets[a.second].a.x)
			- static_cast<CostType> (nets[a.first].subNets[a.second].b.x));
	CostType aH = fabs(static_cast<CostType> (nets[a.first].subNets[a.second].a.y)
			- static_cast<CostType> (nets[a.first].subNets[a.second].b.y));
	pair<CostType, CostType> costA(aW + aH, min(aW, aH) / max(aW, aH));

	CostType bW = fabs(static_cast<CostType> (nets[b.first].subNets[b.second].a.x)
			- static_cast<CostType> (nets[b.first].subNets[b.second].b.x));
	CostType bH = fabs(static_cast<CostType> (nets[b.first].subNets[b.second].a.y)
			- static_cast<CostType> (nets[b.first].subNets[b.second].b.y));
	pair<CostType, CostType> costB(bW + bH, min(bW, bH) / max(bW, bH));

	return costA < costB;
}
