#!/bin/bash

set -ex

benchmark_executable=$1
profile_filename=$2
shift; shift  # removes $1 and $2, now $@ contains all remaining arguments

sudo /usr/local/cuda-13.2/bin/ncu --set full -f -o $profile_filename $benchmark_executable $@
