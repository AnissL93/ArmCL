#!/usr/bin/env sh

set -e

BUILD_PATH=build/build-release
SSHADDR="khadas@khadas-soc"

echo "Compile code"
scons build_dir=build-release Werror=0 -j128 debug=1 asserts=1 neon=1 opencl=1 os=linux arch=arm64-v8a

echo "Copy execute to board $SSHADDR"
sshpass -p "khadas" scp -r execute/ $SSHADDR:~/
echo "Copy so to board"
sshpass -p "khadas" scp $BUILD_PATH/*.so $SSHADDR:~/execute/
echo "Copy graph_expand_conv to board"
sshpass -p "khadas" scp $BUILD_PATH/examples/graph_conv $SSHADDR:~/execute
echo "Execute..."
#sshpass -p "khadas" ssh $SSHADDR "cd ~/execute; export LD_LIBRARY_PATH=~/execute; ./run_all.sh > latency.log 2>&1"
