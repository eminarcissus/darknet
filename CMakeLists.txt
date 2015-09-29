cmake_minimum_required(VERSION 2.8)
project(darknet)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMake")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/pyDarknet)

## Compiler settings

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-sign-compare")

# CUDA
FIND_PACKAGE(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA Version: " ${CUDA_VERSION_STRINGS})
    message(STATUS "CUDA Libararies: " ${CUDA_LIBRARIES})
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    list(APPEND LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY})
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;-std=c++11;-O2;-Xcompiler \"-fPIC\"")
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    add_definitions(-DGPU)
endif()

#BOOST
set(BOOST_ROOT /usr/)
FIND_PACKAGE(Boost REQUIRED python)
FIND_PACKAGE(PythonLibs REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})
include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})

include_directories(src)
cuda_include_directories(src)
set(SRC_DIR src)

FILE(GLOB SRC_FILES ${SRC_DIR}/*.c)
FILE(GLOB CU_SRC_FILES ${SRC_DIR}/*.cu)
LIST(REMOVE_ITEM SRC_FILES "/media/ssd/darknet/src/server.c")
LIST(REMOVE_ITEM SRC_FILES "/media/ssd/darknet/src/old.c")
LIST(REMOVE_ITEM SRC_FILES "/media/ssd/darknet/src/cpu_gemm.c")
LIST(REMOVE_ITEM SRC_FILES "/media/ssd/darknet/src/darknet.c")

message(STATUS "SOURCE FILES: " ${SRC_FILES})
message(STATUS "Boost Libs: " ${Boost_LIBRARIES})

cuda_add_library(dnl SHARED ${SRC_FILES} ${CU_SRC_FILES})
target_link_libraries(dnl ${LIBRARIES})

cuda_add_executable(darknet ${SRC_DIR}/darknet.c)
target_link_libraries(darknet ${LIBRARIES} dnl)

cuda_add_library(pydarknet SHARED  ${SRC_DIR}/pydarknet.cpp)
target_link_libraries(pydarknet ${LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} dnl)