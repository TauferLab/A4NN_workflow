cd ..
echo "Current working directory: $PWD"

export CUBLAS_WORKSPACE_CONFIG=:16:8

export NSGA_NET_PATH=$PWD
export LOWFIVE_PATH=`spack location -i lowfive`
export HENSON_PATH=`spack location -i henson`
export WILKINS_PATH=`spack location -i wilkins`
export LD_LIBRARY_PATH=$LOWFIVE_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HENSON_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$WILKINS_PATH/lib:$LD_LIBRARY_PATH
export HDF5_PLUGIN_PATH=$LOWFIVE_PATH/lib
export HDF5_VOL_CONNECTOR="lowfive under_vol=0;under_info={};"
echo "Finished exporting environment variables..."

cd $NSGA_NET_PATH/nas_search


echo "Running A4NN with Wilkins in file mode"
mpirun -n 3 -l python3 -u ./wilkins-master.py ../configs/wilkins-config-cifar10-example.yaml
