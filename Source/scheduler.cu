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

const unsigned THREADS_PER_BLOCK_X = 32;
const unsigned THREADS_PER_BLOCK_Y = 32;
const unsigned NUM_CONCURRENCY_BINS = 32;
const unsigned SUBNET_COUNT_GPU_THRESHOLD = 20;

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
	cudaDeviceReset();
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
	// std::cout << "Y dim: " << maxY - minY << "\n";
	// std::cout << "X dim: " << maxX - minX << "\n";
	// std::cout << "subnetcount" << subNetCount<< "\n";
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
	// std::cout << "TilesWithinRoutingRegion:\n";
	// for(int j = 0; j < subNetCount; ++j)
	// {
	// 	std::cout << tilesWithinRoutingRegion[j] << " ";
	// }
	// std::cout << std::endl;

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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      std::cout << "GPUassert:  " << cudaGetErrorString(code) << file << line << std::endl;
      if (abort) exit(code);
   }
}

int Scheduler::findConcurrencyGPU(SubNetQueue& subNetsQueue, SubNetQueue& concurrentSubNets, const size_t windowSize)
{
	//GPU implementation of findConcurrencyCPU()
	size_t subNetCount = min(windowSize, subNetsQueue.size());
	if (subNetCount == 0) {
		concurrentSubNets.clear();
		return 0;
	}
	// std::cout << "Subnet Count: " << subNetCount << std::endl;
	// if number of subnets is small enough, run on CPU
	if(subNetCount <= SUBNET_COUNT_GPU_THRESHOLD)
	{
		return this->findConcurrencyCPU(subNetsQueue, concurrentSubNets, windowSize);
	}

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

	std::size_t abSize = sizeof(uint2)*subNetCount;
	uint2* deviceA;
	gpuErrchk(cudaMalloc((void**)&deviceA, abSize));// deallocated
	uint2* deviceB;
	gpuErrchk(cudaMalloc((void**)&deviceB, abSize));// deallocated

	gpuErrchk(cudaMemcpy(deviceA, hostA.data(), abSize, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(deviceB, hostB.data(), abSize, cudaMemcpyHostToDevice));

	const unsigned Nx = maxX - minX + 1;
	const unsigned Ny = maxY - minY + 1;

	IdType* deviceColorTiles; // deallocated
	std::size_t d_pitchColorTiles;
	gpuErrchk(cudaMallocPitch(&deviceColorTiles, &d_pitchColorTiles, sizeof(IdType)*db.yTiles, db.xTiles));
	gpuErrchk(cudaMemset2D(deviceColorTiles, d_pitchColorTiles, 0xFF, sizeof(IdType)*db.yTiles, db.xTiles));

	unsigned* deviceTilesWithinRoutingRegion; // deallocated
	std::size_t d_pitchTilesWithinRoutingRegion;
	gpuErrchk(cudaMallocPitch(&deviceTilesWithinRoutingRegion, &d_pitchTilesWithinRoutingRegion, sizeof(unsigned)*NUM_CONCURRENCY_BINS, subNetCount));
	gpuErrchk(cudaMemset2D(deviceTilesWithinRoutingRegion, d_pitchTilesWithinRoutingRegion, 0, sizeof(unsigned)*NUM_CONCURRENCY_BINS, subNetCount));

	// set up grid sizes

	dim3 dimGrid((Ny + THREADS_PER_BLOCK_Y - 1) / THREADS_PER_BLOCK_Y, (Nx + THREADS_PER_BLOCK_X - 1) / THREADS_PER_BLOCK_X);
	dim3 dimBlock(THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_X);

	colorTiles_noshared<<<dimGrid, dimBlock>>>(deviceColorTiles, deviceA, deviceB, subNetCount, minY, maxY, minX, maxX);
	cudaError err = cudaGetLastError();
	if(cudaSuccess != err)
	{
		std::cout << "Cuda error on histCalc: " << cudaGetErrorString(err) << "\n";
		exit(1);
	}
	// cudaDeviceSynchronize();

	histCalc_noshared<<<dimGrid, dimBlock>>>(deviceTilesWithinRoutingRegion, deviceColorTiles, subNetCount, minY, maxY, minX, maxX, NUM_CONCURRENCY_BINS);
	err = cudaGetLastError();
	if(cudaSuccess != err)
	{
		std::cout << "Cuda error on histCalc: " << cudaGetErrorString(err) << "\n";
		exit(1);
	}
	// cudaDeviceSynchronize();

	unsigned* deviceRetVal; // deallocated
	gpuErrchk(cudaMalloc((void**)&deviceRetVal, sizeof(unsigned)*subNetCount));
	gpuErrchk(cudaMemset(deviceRetVal, 0, sizeof(unsigned)*subNetCount));

	sumHist_noshared<<<1, subNetCount>>>(deviceTilesWithinRoutingRegion, deviceRetVal, subNetCount, NUM_CONCURRENCY_BINS);
	err = cudaGetLastError();
	if(cudaSuccess != err)
	{
		std::cout << "Cuda error on sumHist: " << cudaGetErrorString(err) << "\n";
		exit(1);
	}
	// cudaDeviceSynchronize();

	unsigned* tilesWithinRoutingRegion = (unsigned*) malloc(sizeof(unsigned)*subNetCount); // deallocated

	gpuErrchk(cudaMemcpy(tilesWithinRoutingRegion, deviceRetVal, sizeof(unsigned)*subNetCount, cudaMemcpyDeviceToHost));
	// std::cout << "10" << std::endl;

	// std::cout << "TilesWithinRoutingRegion:\n";
	// for(int j = 0; j < subNetCount; ++j)
	// {
	// 	std::cout << tilesWithinRoutingRegion[j] << " ";
	// }
	// std::cout << std::endl;

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

	// deallocate
	free(hostA);
	free(hostB);
	gpuErrchk(cudaFree(deviceA));
	gpuErrchk(cudaFree(deviceB));

	gpuErrchk(cudaFree(deviceColorTiles));
	gpuErrchk(cudaFree(deviceTilesWithinRoutingRegion));
	gpuErrchk(cudaFree(deviceRetVal));
	free(tilesWithinRoutingRegion);

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
