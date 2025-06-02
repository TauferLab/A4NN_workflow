#!/bin/bash

export SPACKENV=nas-workflow-env
export YAML=$PWD/env.yaml

# create spack environment
echo "creating spack environment $SPACKENV"
spack env deactivate > /dev/null 2>&1
spack env remove -y $SPACKENV > /dev/null 2>&1
spack env create $SPACKENV $YAML

# activate environment
echo "activating spack environment"
spack env activate $SPACKENV

spack add lowfive

spack develop wilkins@master build_type=Debug
spack add wilkins

spack add henson+python+mpi-wrappers

# install
echo "installing dependencies in environment"
spack install       # install the rest

# reset the environment (workaround for spack behavior)
spack env deactivate
spack env activate $SPACKENV

# set build flags
echo "setting flags for building moab-workflow"
export LOWFIVE_PATH=`spack location -i lowfive`
export HENSON_PATH=`spack location -i henson`
export WILKINS_PATH=`spack location -i wilkins`

# set LD_LIBRARY_PATH
echo "setting flags for running moab-workflow"
export LD_LIBRARY_PATH=$LOWFIVE_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HENSON_PATH/lib:$LD_LIBRARY_PATH

export HDF5_PLUGIN_PATH=$LOWFIVE_PATH/lib
export HDF5_VOL_CONNECTOR="lowfive under_vol=0;under_info={};"

#needed for the NAS
export NSGA_NET_PATH=$PWD
