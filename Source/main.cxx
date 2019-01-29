/*
 * main.cxx
 *
 *  Created on: Jul 9, 2010
 *      Author: yiding
 */
#include <iostream>

#include "database.h"
#include "utility.h"
#include "commandline.h"
#include "scheduler.h"

using std::cout;
using std::endl;

int main(int argc, const char **argv)
{
	// Read parameters
	CommandLine commands(argc, argv);

	if (commands.helpOnly) {
		commands.printHelp(argc, argv);
		exit(0);
	}

	// Print out version info
	cout << _PROJECT_ << " (" << sizeof(void*) * 8 << "-bit) "
			<< __DATE__ << " @ " << __TIME__ << endl;
#ifdef __INTEL_COMPILER
	cout << "Compiled with Intel C++ Compiler " << __INTEL_COMPILER;
#else
#ifdef __GNUG__
	cout << "Compiled with g++ " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#endif
#endif
	cout << endl << __NVCCVER__ << endl << endl;

	//print user commands
	cout << "== User Parameters ==" << endl;
	commands.printCommands();

	// Initialize database
	cout << "== Initialize Data ==" << endl;
	DB db(commands);

	// Start
	cout << "== Start Computation ==" << endl;
	Scheduler engine(db, commands);
	engine.run();

	// Output results
	if (commands.allowOutput) {
		cout << "== Save Results ==" << endl;
		db.writeOutput();
	}

	return 0;
}
