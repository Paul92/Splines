cmake_minimum_required(VERSION 3.4...3.18)
project(splines)

add_subdirectory(pybind11)

find_package(Eigen3 REQUIRED)

set(MKLROOT "/opt/intel/mkl/")
set(INTEL_MKL_INCLUDE_DIRS "${MKLROOT}/include")

pybind11_add_module(splines src/main.cpp src/warps.cpp src/bbs.cpp)


target_include_directories(splines PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_include_directories(splines PRIVATE ${INTEL_MKL_INCLUDE_DIRS})

#target_link_libraries(splines PUBLIC "-lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core")
target_link_libraries(splines PUBLIC "-lgomp -lpthread -lm -ldl -lgmp -lglog -lpython3.9 -lumfpack -lamd -lcholmod -lsuitesparseconfig -lbtf -lcamd -lccolamd -lcxsparse -lklu -lldl -lrbio -lsliplu -lspqr")

target_link_directories(splines PUBLIC "${MKLROOT}/lib/intel64")

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(splines
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})


add_executable(test tests/test.cpp src/bbs.cpp src/warps.cpp src/warps.h)

target_include_directories(test PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(test PUBLIC "-lgomp -lpthread -lm -ldl -lgmp -lglog -lpython3.9 -lumfpack -lamd -lcholmod -lsuitesparseconfig -lbtf -lcamd -lccolamd -lcxsparse -lklu -lldl -lrbio -lsliplu -lspqr")

target_compile_definitions(splines PRIVATE NTHREADS=8)
target_compile_definitions(test PRIVATE NTHREADS=8)
