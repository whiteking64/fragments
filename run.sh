# template shell script for git bisect for pytorch/glow
status=0

cd /root/work/glow/build_Debug
ninja clean
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DGLOW_WITH_BUNDLES=ON ..
ninja all
status=$?

if [ $status -eq 125 ] || [ $status -gt 127 ]; then
  status=1
fi
exit $status
