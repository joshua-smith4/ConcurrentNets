#!/bin/sh

HOST_ARCH=`uname -m`

case $HOST_ARCH in
    i*86) 
        ARCH=x86
	;;
    x86_64)
        ARCH=x86_64
	;;
    amd64)
        ARCH=x86_64
	;;
    sun4u|sun4v)
        ARCH=v9
	;;
    *)	
        ARCH=x86
	;;
esac

echo $ARCH
exit 0
