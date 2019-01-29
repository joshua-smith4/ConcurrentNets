/*
 * scheduler.h
 * Scheduler is the main object that performs the tile coloring algorithm,
 * and extracts concurrent subnets.
 *
 *  Created on: Apr 14, 2011
 *      Author: yiding
 */

#ifndef SCHEDULER_H_
#define SCHEDULER_H_

#include <vector>
#include <cuda_runtime.h>	//Include for declaration of uint2 type

using std::vector;

#include "common.h"
#include "database.h"
#include "utility.h"

class Scheduler
{
private:
	IdType * hostTiles;


	int findConcurrencyGPU(SubNetQueue& subNetsQueue, SubNetQueue& concurrentSubNets, const size_t windowSize);
	int findConcurrencyCPU(SubNetQueue& subNetsQueue, SubNetQueue& concurrentSubNets, const size_t windowSize);

protected:
	const CommandLine& params;
	DB& db;
public:
	Scheduler(DB& _db, const CommandLine& _params);

	int findCUDADevice(vector<int>& deviceList, const unsigned minMajorVer, const unsigned minMinerVer);
	int run();

	~Scheduler();
};

class CompareByBox
{
	const vector<Net> &nets;
public:
	CompareByBox(const vector<Net> &n) :
		nets(n)
	{
	}

	bool operator()(const SubNetIdType &a, const SubNetIdType &b) const;
};

#endif /* SCHEDULER_H_ */
