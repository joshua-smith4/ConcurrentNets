/*
 * common.h
 * This header defines basic global data types in the project.
 *
 *  Created on: Jul 13, 2010
 *      Author: yiding
 */

#ifndef _COMMON_H_
#define _COMMON_H_

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <list>

using std::pair;
using std::vector;
using std::list;
using std::numeric_limits;
using std::make_pair;

typedef unsigned int CoordType;
typedef unsigned int IdType;
typedef unsigned short CapType;
typedef unsigned int SizeType;
typedef float CostType;

typedef pair<IdType, IdType> SubNetIdType;

typedef vector<SubNetIdType> SubNetQueue;

const IdType NO_ID = numeric_limits<IdType>::max();

class Point
{
public:
	CoordType x, y, z;

	Point() :
		x(0), y(0), z(0)
	{
	}
	Point(CoordType X, CoordType Y, CoordType Z) :
		x(X), y(Y), z(Z)
	{
	}
	Point(const Point &orig) :
		x(orig.x), y(orig.y), z(orig.z)
	{
	}

	const Point &operator=(const Point &assign)
	{
		x = assign.x;
		y = assign.y;
		z = assign.z;
		return *this;
	}
};

inline bool operator==(const Point &a, const Point &b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline bool operator!=(const Point &a, const Point &b)
{
	return a.x != b.x || a.y != b.y || a.z != b.z;
}

inline bool operator<(const Point &a, const Point &b)
{
	return a.x < b.x || (a.x == b.x && a.y < b.y) || (a.x == b.x && a.y == b.y && a.z < b.z);
}

class SubNet
{
public:
	Point a, b;

	SubNet() {}
	SubNet(const SubNet &orig) :
		a(orig.a), b(orig.b)
	{
	}
	inline SizeType size(void)
	{
		return (abs(static_cast<int>(a.x - b.x)) + 1) * (abs(static_cast<int>(a.y - b.y)) + 1);
	}
	inline SizeType width(void)
	{
		return (abs(static_cast<int>(a.x - b.x)) + 1);
	}
	inline SizeType height(void)
	{
		return (abs(static_cast<int>(a.y - b.y)) + 1);
	}
};

class Net
{
public:
	SizeType numSegments, numVias;
	CapType minWidth;
	vector<Point> pins;
	vector<SubNet> subNets;

	Net() :
		numSegments(0), numVias(0), minWidth(0)
	{
	}
	Net(const Net &orig) :
		numSegments(orig.numSegments), numVias(orig.numVias), minWidth(orig.minWidth),
				pins(orig.pins), subNets(orig.subNets)
	{
	}
};

#endif /* ROUTERCOMMON_H_ */
