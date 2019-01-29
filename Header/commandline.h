/*
 * Commandline.h
 *
 *  Created on: Jul 13, 2010
 *      Author: yiding
 *
 */

#ifndef COMMANDLINE_H_
#define COMMANDLINE_H_

#include <string>
#include "paramproc.h"

using std::string;

class CommandLine
{
public:
	CommandLine(int argc, const char **argv);
	CommandLine();

	bool helpOnly;
	string inputFileName;		// input file name
	string outputFileName;		// output results file name

	bool allowGPU;				//turn it off when you don't want to use GPU
	unsigned schedulingWindow;	//The window size to find concurrent subnets

	/*---------------------More command line options can be defined here-------------------------------------*/
								//examples
	bool printProgress;			//turn progress off when piping the screen to a file
	bool allowOutput;			//turn it off if you don't want any output

	/*-------------------------------------command line options end here-------------------------------------*/

	// member functions
	void printHelp(int argc, const char **argv) const;
	// print router parameters
	void printCommands() const;
};

#endif /* COMMANDLINE_H_ */
