#!/bin/sh

mkdir -p model/ss34
cd model/ss34
curl -O 'https://dl.dropboxusercontent.com/s/66q2ts4q730pvpo/ss34.tar.gz'
tar -xvf ss34.tar.gz
cd ../..
python3 ss34.py test $1 $2 $3
