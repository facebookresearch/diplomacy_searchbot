cmake_minimum_required(VERSION 3.10)
project(dipcc VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# For pybind11 to work on Linux.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out)

# Declare dependencies
find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(pybind11 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIR})

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -pg")

# Compile dipcc
file(GLOB SOURCES "dipcc/cc/*.cc")
add_library(dipcc ${SOURCES})
target_link_libraries(dipcc glog pthread)

# Compile pybind
file(GLOB PY_SOURCES "dipcc/pybind/*.cc")
pybind11_add_module(pydipcc ${PY_SOURCES} ${SOURCES})
set_target_properties(pydipcc PROPERTIES LIBRARY_OUTPUT_DIRECTORY "dipcc/python/")
target_link_libraries(pydipcc PRIVATE glog pthread)

# Compile profiling
IF(CMAKE_BUILD_TYPE MATCHES RELEASE)
    add_executable(profile_dipcc dipcc/profiling/profile.cc ${SOURCES})
    target_link_libraries(profile_dipcc dipcc glog pthread)

    add_executable(profile_dipcc_late dipcc/profiling/profile_late.cc ${SOURCES})
    target_link_libraries(profile_dipcc_late dipcc glog pthread)
ENDIF(CMAKE_BUILD_TYPE MATCHES RELEASE)

# Compile tests
IF(NOT DEFINED ENV{SKIP_TESTS})
    enable_testing()
    add_subdirectory(${CMAKE_SOURCE_DIR}/../thirdparty/github/fairinternal/postman/third_party/grpc/third_party/googletest ${CMAKE_CURRENT_BINARY_DIR}/googletest)
    include(GoogleTest)
    file(GLOB TEST_SRC "dipcc/tests/*.cc")
    add_executable(test_dipcc ${TEST_SRC} ${SOURCES})
    target_link_libraries(test_dipcc dipcc gtest gtest_main)
    gtest_add_tests(TARGET test_dipcc AUTO)
ENDIF(NOT DEFINED ENV{SKIP_TESTS})
