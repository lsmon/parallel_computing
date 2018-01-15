
file(GLOB hw2_cu *.cu)
file(GLOB hw2_hdr *.hpp *.h *.cuh)
set(HW2_SRC main.cpp reference_calc.cpp compare.cpp)
CUDA_ADD_EXECUTABLE(HW2 ${HW2_SRC} ${hw2_hdr} ${hw2_cu})