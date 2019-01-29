/*
 * utility.h
 * This header defines some useful tools for the program.
 *
 *  Created on: Mar 18, 2011
 *      Author: yiding
 */

#ifndef UTILITY_H_
#define UTILITY_H_

#include <vector>
#include <ctime>
#include <time.h>
#include <iostream>

#include <unistd.h>
#include <limits.h>

using std::vector;

/*  An efficient stopwatch that reads wall clock time. */
class WallClockTimer
{
public:
	WallClockTimer(void) :
		onGoing(false), currentIdx(0)
	{
		sysconf(_SC_MONOTONIC_CLOCK);
		clock_gettime(CLOCK_MONOTONIC, &startTime);
		clock_gettime(CLOCK_MONOTONIC, &stopTime);

		interval.push_back(0.0);
	}

	WallClockTimer(const WallClockTimer& orig) :
		startTime(orig.startTime), stopTime(orig.stopTime), onGoing(orig.onGoing),
				currentIdx(orig.currentIdx), interval(orig.interval)
	{
	}

	/* Often used functions */
	void start(void);
	void stop(void);
	void pause(void);
	double reading(void) const;

	/* Rarely used functions */
	double readings(unsigned index) const;
	void checkpoint(void);
private:
	//timer variables
	struct timespec startTime;
	struct timespec stopTime;
	bool onGoing;
	int currentIdx;
	vector<double> interval;
};

/* A visual percentage progress printer */
class ProgressIndicator
{
public:
	ProgressIndicator(int _begin, int _end, bool _en);
	void print(int curr, const char * title);
	inline void turnOff(void) {enabled = false;}	// turn off the progress post
	inline void turnOn(void) {enabled = true;}		// turn on the progress post
private:
	int begin;
	unsigned length;
	unsigned latestPercentage;
	bool enabled;
};

#endif /* UTILITY_H_ */
