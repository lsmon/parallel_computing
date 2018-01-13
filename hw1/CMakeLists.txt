############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files

file(GLOB hw1_cu *.cu)
file(GLOB hw1_hdr *.hpp *.h *.cuh)
set(HW1_SRC main.cpp reference_calc.cpp compare.cpp)
CUDA_ADD_EXECUTABLE(HW1 ${HW1_SRC} ${hw1_hdr} ${hw1_cu})