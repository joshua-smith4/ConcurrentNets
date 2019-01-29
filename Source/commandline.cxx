/*
 * Commandline.cxx
 *
 *  Created on: Jul 13, 2010
 *      Author: yiding
 */

#include <iostream>
#include <string>
#include <assert.h>

#include "commandline.h"

#define DEFAULT_SCHEDULING_WINDOW	256
#define DEFAULT_OUTPUT_FILE			"output.txt"

using std::cout;
using std::endl;

CommandLine::CommandLine()
	: helpOnly(false), inputFileName(""), outputFileName(DEFAULT_OUTPUT_FILE)
{ }

CommandLine::CommandLine(int argc, const char **argv)
	: helpOnly(false), inputFileName(""), outputFileName(DEFAULT_OUTPUT_FILE)
{
	/* Define Param objects to catch command line options
	 * Param objects includes:
	 * BoolParam      : boolean option
	 * StringParam    : string option
	 * UnsignedParam  : unsigned integer option
	 * IntParam       : signed integer option, '-' sign can be recognized
	 * DoubleParam    : double floating point option.
	 * */
	BoolParam helpShort("h", argc, argv);
	BoolParam helpLong("help", argc, argv);

	NoParams noParams(argc, argv);

	StringParam argInputFile("f", argc, argv);
	StringParam argOutputFile("o", argc, argv);

	BoolParam argAllowGPU("gpu", argc, argv);
	UnsignedParam argSchedulingWindow("schwin", argc, argv);

	/*---------------------Additional Param objects can be defined here-------------------------------------*/
	//examples
	BoolParam argPrintProgress("noprint", argc, argv);
	BoolParam argAllowOutput("nooutput", argc, argv);

	/*---------------------------Param objects definition ends here-----------------------------------------*/

	/* Use Param objects to control the values of the command line options
	 * If the Param object is found then initialize the command line option with
	 * the detected value; if not, give default value.
	 */
	if (helpShort.found() || helpLong.found())
	{
		//when no parameter is found we only print 'help screen'
		helpOnly = true;
	}

	//control the input file option
	if (argInputFile.found()) {
		inputFileName = argInputFile;
	}

	if (noParams.found() || !argInputFile.found()) {
		if (!helpOnly) {
			cout << "Please specify input file name." << endl;
			helpOnly = true;
		}
	}

	//control the output file option
	if (argOutputFile.found()) {
		outputFileName = argOutputFile;
	}

	//control the allowGPU option
	if (argAllowGPU.found()) {
		allowGPU = true;
	}
	else {
		allowGPU = false;
	}

	//control the scheduling window size option
	if (argSchedulingWindow.found()) {
		schedulingWindow = argSchedulingWindow;
		if (schedulingWindow == 0) {
			cout << "Warning: scheduling window must be positive. Using default value." << endl;
			schedulingWindow = DEFAULT_SCHEDULING_WINDOW;
		}
	}
	else {
		schedulingWindow = DEFAULT_SCHEDULING_WINDOW;
	}

	/*---------------------Additional command line options are controlled here-------------------------------------*/
	//examples
	//control the printProgress option
	if (argPrintProgress.found()) {
		printProgress = false;
	}
	else {
		printProgress = true;
	}

	//control the allowOutput option
	if (argAllowOutput.found()) {
		allowOutput = false;
	}
	else {
		allowOutput = true;
	}

	/*---------------------Additional command line options ends here-----------------------------------------------*/

}

void CommandLine::printHelp(int argc, const char **argv) const
{
	cout << "-f filename : Specify input file name.\n" <<
			"-o filename : Specify output file name. Default value: " << DEFAULT_OUTPUT_FILE << ".\n" <<
			"-noprint    : Turn off progress printing. This is useful when piping the screen output to a file. \n" <<
			"-gpu        : Turn on GPU computing. \n" <<
			"-schwin     : Specify the size of scheduling window. Scheduling window defines the number of subnets used to color the tiles in each iteration. A wide window might extract more concurrent subnets, but will significantly slow down the process. Default value: " << DEFAULT_SCHEDULING_WINDOW << ".\n" <<
			"-nooutput   : Turn off writing results to file at the end of execution. \n" <<
			endl;
}

void CommandLine::printCommands() const
{
	cout << "Input file:            " << inputFileName << endl;
	if (allowOutput) {
		cout << "Output file:           " << outputFileName << endl;
	}
	cout << "Computing platform:    " << (allowGPU ? "GPU" : "CPU") << endl;
	cout << "Scheduling window:     " << schedulingWindow << endl;
	cout << "Print progress:        " << (printProgress ? "Enabled" : "Disabled") << endl << endl;
}
