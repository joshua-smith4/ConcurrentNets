/*
 * utility.cxx
 *
 *  Created on: Mar 18, 2011
 *      Author: yiding
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using std::cout;
using std::flush;
using std::endl;

#include "utility.h"

/*
 * Start the timer. Resets timer if it is stopped; Resumes
 * the timer if it is paused.
 */
void WallClockTimer::start(void)
{
	if (!onGoing)
		interval[currentIdx] = 0.0;
	clock_gettime(CLOCK_MONOTONIC, &startTime);
}

/*
 * Stop and reset timer
 */
void WallClockTimer::stop(void)
{
	clock_gettime(CLOCK_MONOTONIC, &stopTime);
	interval[currentIdx] = stopTime.tv_sec - startTime.tv_sec;
	interval[currentIdx] += (stopTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;
	onGoing = false;
}

/*
 * Pause the timer, then can be restarted using start() function
 */
void WallClockTimer::pause(void)
{
	clock_gettime(CLOCK_MONOTONIC, &stopTime);
	interval[currentIdx] += stopTime.tv_sec - startTime.tv_sec;
	interval[currentIdx] += (stopTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;
	onGoing = true;
}

/*
 * Add the current time as a checkpoint into the current interval
 * slot, and move the slot index to the next one.
 * The timer is still running, but the next time timer stops, the
 * interval will be recorded into the next interval slot. Just like
 * clicking checkpoint on a stopwatch.
 */
void WallClockTimer::checkpoint(void)
{
	clock_gettime(CLOCK_MONOTONIC, &stopTime);
	interval[currentIdx] = stopTime.tv_sec - startTime.tv_sec;
	interval[currentIdx] += (stopTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;
	interval.push_back(0.0);
	currentIdx++;
	onGoing = true;
}

/*
 * return the recorded interval saved in a slot pointed by an index.
 * There are multiple indices only when checkpoint() is called.
 * Otherwise the interval is recorded in the 0 slot.
 */
double WallClockTimer::readings(unsigned index) const
{
	if (index < interval.size())
		return interval[index];
	else
		return 0.0;
}

/*
 * return the recorded interval from the first slot (0 slot).
 */
double WallClockTimer::reading(void) const
{
	return interval[0];
}

/* Constructor of ProgressIndicator object
 * params:
 * _begin: the starting index (e.g. of a loop)
 * _end:   the ending index (e.g. of a loop)
 * _en:    a boolean to indicate whether to post progress or not
 *         true - turn on progress post
 *         false - turn off the post, useful when piping the output
 *         to a file, or for debugging purposes.
 */
ProgressIndicator::ProgressIndicator(int _begin, int _end, bool _en)
{
	enabled = _en;
	length = abs(_end - _begin);
	begin = _begin;
	latestPercentage = 100;
}

/* Print a message and the current progress
 * params:
 * curr:  the current index (e.g. of a loop)
 * title: the C string message that is printed with the progress
 *        (e.g. "Initializing")
 */
void ProgressIndicator::print(int curr, const char * title)
{
	if (length == 0) {
		return;
	}
	if (enabled) {
		unsigned currProgress = abs(curr - begin) + 1;
		unsigned currPercentage = currProgress * 100 / length;
		if (latestPercentage != currPercentage) {
			latestPercentage = currPercentage;
			if (currProgress >= length)
				cout << title << " ... Done." << endl << flush;
			else
				cout << title << " ... " << latestPercentage << "%\r" << flush;
		}
	}
	else {
		if (abs(curr - begin) + 1 == length)
			cout << title << " ... Done." << endl << flush;
	}
}
