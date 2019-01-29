/*
 * DB.cxx
 *
 *  Created on: Jul 16, 2010
 *      Author: yiding
 */

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <assert.h>

#include "database.h"
#include "common.h"
#include "commandline.h"
#include "utility.h"

using std::ifstream;
using std::ofstream;
using std::string;
using std::sort;
using std::cout;
using std::endl;
using std::flush;
using std::min;
using std::max;
using std::swap;

static bool expect(ifstream& inputFile, const char * expecting) {
	string buff;

	inputFile >> buff;
	if (buff != string(expecting)) {
		cout << "Parsing error. Expecting \"" << expecting << "\" but read \"" << buff << endl;
		exit(0);
	}
	return true;
}

DB::DB(CommandLine &commands) :
	params(commands)
{
	//Read the benchmark from the input file
	readInputFile(params.inputFileName.c_str());

	concurrentSubnets.clear();

	// Print out routing info
	printGridInfo();
}

void DB::readInputFile(const char * fileName) {
	ifstream inputFile(fileName);
	string buff;

	if (!inputFile.good()) {
		cout << "Cannot open \"" << fileName << "\". Check the file path." << endl;
		exit(0);
	}
	else {
		cout << "Reading from file \"" << fileName << "\"." << endl;
	}

	// Read grid
	//-------------------------------------------------------------------------
	expect(inputFile, "grid");
	inputFile >> xTiles >> yTiles >> numLayers;
	zTiles = 2;	// FIXME: hardcoded to to layer assignment

	// Read capacity parameters
	expect(inputFile, "vertical");
	expect(inputFile, "capacity");
	for (int i = 0; i < numLayers; i++) {
		CapType cap;
		inputFile >> cap;
		vertiCaps.push_back(cap);
	}

	expect(inputFile, "horizontal");
	expect(inputFile, "capacity");
	for (int i = 0; i < numLayers; i++) {
		CapType cap;
		inputFile >> cap;
		horizCaps.push_back(cap);
	}

	expect(inputFile, "minimum");
	expect(inputFile, "width");
	for (int i = 0; i < numLayers; i++) {
		CapType cap;
		inputFile >> cap;
		minWidths.push_back(cap);
	}

	expect(inputFile, "minimum");
	expect(inputFile, "spacing");
	for (int i = 0; i < numLayers; i++) {
		CapType cap;
		inputFile >> cap;
		minSpacings.push_back(cap);
	}

	expect(inputFile, "via");
	expect(inputFile, "spacing");
	for (int i = 0; i < numLayers; i++) {
		CapType cap;
		inputFile >> cap;
		viaSpacings.push_back(cap);
	}

	inputFile >> minX >> minY >> tileWidth >> tileHeight;
	halfWidth = static_cast<SizeType> (trunc(0.5 * tileWidth));
	halfHeight = static_cast<SizeType> (trunc(0.5 * tileHeight));

	//-------------------------------------------------------------------------
	// Read nets
	//-------------------------------------------------------------------------
	numSubNets = 0;
	expect(inputFile, "num");
	expect(inputFile, "net");
	inputFile >> numNets;
	nets.reserve(numNets);
	ProgressIndicator progress(0, numNets, params.printProgress);

	for (SizeType i = 0; i < numNets; i++) {
		string name;
		IdType netId;
		SizeType numPins;
		Net newNet;

		// print out the progress of this loop
		progress.print(i, "Reading Nets");

		inputFile >> name >> netId >> numPins >> newNet.minWidth;

		for (SizeType j = 0; j < numPins; j++) {
			double pinX, pinY;
			CoordType pinZ;

			inputFile >> pinX >> pinY >> pinZ;

			Point pin;
			pin.x = static_cast<CoordType> (floor((pinX - minX) / tileWidth));
			pin.y = static_cast<CoordType> (floor((pinY - minY) / tileHeight));
			pin.z = pinZ - 1;

			newNet.pins.push_back(pin);
		}

		sort(newNet.pins.begin(), newNet.pins.end());
		newNet.pins.erase(unique(newNet.pins.begin(), newNet.pins.end()), newNet.pins.end());

		if (newNet.pins.size() > 1) {
			nets.push_back(newNet);
			netNames.push_back(name);
			netIds.push_back(netId);

			//MST decomposition
			numSubNets += doMST(newNet);
		}
	}

	numNets = nets.size();

	inputFile.close();
}

/* perform MST decomposition for multi-pin nets */
SizeType DB::doMST(Net& newNet) {
	SizeType numMSTSubNets = 0;
	if (newNet.pins.size() > 1) {
		// build the MST tree
		vector<pair<Point, Point> > mstTree;
		vector<unsigned> distanceToTree(newNet.pins.size(), UINT_MAX);
		vector<bool> inTree(newNet.pins.size(), false);
		vector<unsigned> parent(newNet.pins.size(), UINT_MAX);

		unsigned currentPin = 0, nextPin = UINT_MAX;

		for (unsigned j = 0; j < newNet.pins.size() - 1; ++j) {
			inTree[currentPin] = true;
			unsigned bestDistance = UINT_MAX;
			for (unsigned k = 0; k < newNet.pins.size(); ++k) {
				if (inTree[k])
					continue;

				unsigned thisDistance = max(newNet.pins[k].x, newNet.pins[currentPin].x) - min(newNet.pins[k].x,
						newNet.pins[currentPin].x) + max(newNet.pins[k].y, newNet.pins[currentPin].y) - min(
						newNet.pins[k].y, newNet.pins[currentPin].y);

				if (thisDistance < distanceToTree[k]) {
					distanceToTree[k] = thisDistance;
					parent[k] = currentPin;
				}
				if (distanceToTree[k] < bestDistance) {
					bestDistance = distanceToTree[k];
					nextPin = k;
				}

			}
			mstTree.push_back(make_pair(newNet.pins[nextPin], newNet.pins[parent[nextPin]]));
			currentPin = nextPin;
		}

		// add in the MST tree
		for (unsigned j = 0; j < mstTree.size(); ++j) {
			SubNet stemp;
			stemp.a = mstTree[j].first;
			stemp.b = mstTree[j].second;
			newNet.subNets.push_back(stemp);
			numMSTSubNets++;
		}

		nets.push_back(newNet);
	}
	return numMSTSubNets;
}

void DB::printGridInfo() {
	cout << "Grid size " << xTiles << " x " << yTiles << endl;
	cout << "Layers " << zTiles << endl;
	cout << "Read " << numNets << " nets" << endl;
	cout << "Decomposed into " << numSubNets << " subnets" << endl << endl;
}

void DB::writeOutput(void)
{
	ofstream outfile(params.outputFileName.c_str());

	if (!outfile.good()) {
		cout << "Could not open `" << params.outputFileName.c_str() << "' for writing." << endl;
		return;
	}

	cout << "Writing to \"" << params.outputFileName.c_str() << "\" ..." << flush;

	// Write contents
	for (list<SubNetQueue>::iterator queue = concurrentSubnets.begin(); queue != concurrentSubnets.end(); queue++) {
		outfile << queue->size() << ";";
		for (SubNetQueue::iterator snet = queue->begin(); snet != queue->end(); snet++) {
			outfile << "(" << nets[snet->first].subNets[snet->second].a.x << "," << nets[snet->first].subNets[snet->second].a.y << ")-";
			outfile << "(" << nets[snet->first].subNets[snet->second].b.x << "," << nets[snet->first].subNets[snet->second].b.y << ");";
		}
		outfile << endl;
	}

	outfile.close();
	cout << " Done." << endl;
}

