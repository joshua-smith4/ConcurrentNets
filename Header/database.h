/*
 * database.h
 * This header defines a database that stores all input and output data.
 *
 *  Created on: Jul 16, 2010
 *      Author: yiding
 */

#ifndef DB_H_
#define DB_H_

#include "common.h"
#include "commandline.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <pthread.h>

using std::vector;

class DB
{
public:
	SizeType xTiles, yTiles, zTiles;
	SizeType numLayers;
	SizeType numNets, numSubNets;

	vector<CapType> vertiCaps, horizCaps, minWidths, minSpacings, viaSpacings;
	vector<Net> nets;
	IdType minX, minY;
	SizeType tileWidth, tileHeight, halfWidth, halfHeight;
	vector<string> netNames;
	vector<IdType> netIds;

	CommandLine& params;

	list<SubNetQueue> concurrentSubnets;

	DB(CommandLine& commands);

	void printGridInfo(void);
	void writeOutput(void);

private:
	void readInputFile(const char *fileName);
	SizeType doMST(Net& newNet);
};

#endif /* DB_H_ */
