/**************************************************************************
***    
*** Copyright (c) 1995-2000 Regents of the University of California,
***               Andrew E. Caldwell, Andrew B. Kahng and Igor L. Markov
*** Copyright (c) 2000-2010 Regents of the University of Michigan,
***               Saurabh N. Adya, Jarrod A. Roy, David A. Papa and
***               Igor L. Markov
***
***  Contact author(s): abk@cs.ucsd.edu, imarkov@umich.edu
***  Original Affiliation:   UCLA, Computer Science Department,
***                          Los Angeles, CA 90095-1596 USA
***
***  Permission is hereby granted, free of charge, to any person obtaining 
***  a copy of this software and associated documentation files (the
***  "Software"), to deal in the Software without restriction, including
***  without limitation 
***  the rights to use, copy, modify, merge, publish, distribute, sublicense, 
***  and/or sell copies of the Software, and to permit persons to whom the 
***  Software is furnished to do so, subject to the following conditions:
***
***  The above copyright notice and this permission notice shall be included
***  in all copies or substantial portions of the Software.
***
*** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
*** EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
*** OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
*** IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
*** CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
*** OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
*** THE USE OR OTHER DEALINGS IN THE SOFTWARE.
***
***
***************************************************************************/


// Created: June 15, 1997 Igor Markov

// 970622 ilm    added NOPARAM
// 970820 ilm    moved enum ParamType inside class Param
//               added new ctor for use with NOPARAM
//               allowed for +option as well as -option
//               added member bool on() to tell if it was +option
//               added abkfatal(found(),...) when reading param, value
// 970824 ilm    changed the uninitialized value for int to MAX_INT
// 980313 ilm    fixed const-correctness

// Modified: Sep 26, 2010 Yiding Han

// 110926	yiding	removed abkassert macros
//					added local _fatal macro

#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif

#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <stdio.h>

#define _fatal(assertion, msg) if (!(assertion)) { printf(msg); printf("\n"); exit(0); }

#include "paramproc.h"
static char _uninitialized[]="Uninitialized";

Param::Param(Type pt,int argc, const char * const argv[])
: _b(false), _on(false), _i(INT_MAX), _u(unsigned(-1)), _d(-1.29384756657), 
  _s(_uninitialized), _pt(pt), _key("")
{
  _fatal(pt==NOPARAM," This constructor can only work with Param:NOPARAM\n");
  (void)argv; // please compiler 
   
  _b=(argc<2?true:false);
  return;
} 



Param::Param(const char * key, Type pt, int argc, const char * const argv[]) :
	_b(false), _on(false), _i(-1), _u(unsigned(-1)), _d(-1.23456), _s(_uninitialized), _pt(pt),
			_key(key)
{
	_fatal(strlen(_key)>0," Zero length key for command line parameter");

	int n = 0;
	if (_pt == NOPARAM) {
		if (argc < 2)
			_b = true;
		else
			_b = false;
		return;
	}
	while (++n < argc && !found()) {
		if (argv[n][0] == '-' || argv[n][0] == '+') {
			const char * start = argv[n] + 1;
			if (argv[n][0] == '-') {
				if (argv[n][1] == '-')
					start++;
			}
			else
				_on = true;

			if (strcasecmp(start, _key) == 0) {
				_b = true;
				if (n + 1 < argc) {
					switch (_pt) {
						case BOOL:
							break;
						case INT:
							_i = atoi(argv[n + 1]);
							break;
						case UNSIGNED:
							_u = strtoul(argv[n + 1], (char**) NULL, 10);
							break;
						case DOUBLE:
							_d = atof(argv[n + 1]);
							break;
						case STRING:
							_s = argv[n + 1];
							break;
						default:
							break;
							_fatal(0," Unknown command line parameter");
					}
				}
			}
		}
	}
}

bool      Param::found()       const
{ return _b; }

bool Param::on() const // true for +option, false otherwise
{
	_fatal(found()," Parameter not found: you need to check for this first\n");
	return _on;
}

int Param::getInt() const
{
	_fatal(_pt==INT," Parameter is not INT ");
	_fatal(found()," Parameter not found: you need to check for this first\n");
	return _i;
}

unsigned Param::getUnsigned() const
{
	_fatal(_pt==UNSIGNED," Parameter is not UNSIGNED ");
	_fatal(found(), " UNSIGNED Parameter not found: you need to check for this first\n");
	return _u;
}

double Param::getDouble() const
{
	_fatal(_pt==DOUBLE," Parameter is not DOUBLE ");
	_fatal(found(), " DOUBLE parameter not found: you need to check for this first\n");
	return _d;
}

const char * Param::getString() const
{
	_fatal(_pt==STRING," Parameter is not STRING");
	_fatal(found(), " STRING parameter not found: you need to check for this first\n");
	return _s;
}
