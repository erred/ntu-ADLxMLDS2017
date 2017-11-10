#!/bin/sh

mkdir -p model/ss34
cd model/ss34
curl 'https://dl.dropboxusercontent.com/s/66q2ts4q730pvpo/ss34.tar.gz' > ss34.tar.gz
tar -xvf ss34.tar.gz
cd ../..
python3 ss34.py test-special $1 $2
