#
# CMake build system
# This file is part of MLIP
#
# Contributors:
#    Yaser Afshar
#    Ryan S. Elliott
#
cmake_minimum_required(VERSION 3.10)

# Define main project
project(MLIP LANGUAGES CXX C Fortran VERSION 0.2)

# Project authors
set(AUTHOR "Alexander Shapeev, Evgeny Podryabinkin, Konstantin Gubaev, and Ivan Novikov")
set(AUTHOR_DETAILS "")
set(DESCRIPTION "MLIP is a software for Machine Learning Interatomic Potentials.")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)
  message(STATUS "Setting the build type to \"RelWithDebInfo\" as none was specified.")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Set the build type in this build tree." FORCE)
  # Possible values of build type in MLIP for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      "${CMAKE_SOURCE_DIR}/cmake"
                      "${CMAKE_SOURCE_DIR}/cmake/Modules")

# Project build options
option(WITH_MPI "Build an MPI version" ON)
if(WITH_MPI)
  find_package(MPI QUIET) # sets MPI_FOUND=TRUE if successful
  if(MPI_FOUND)
    message(STATUS "Found MPI: TRUE (found version \"${MPI_CXX_VERSION}\")")
  endif(MPI_FOUND)
endif(WITH_MPI)
option(WITH_SELFTEST "Enable 'self-testing' implementation" ON)
option(WITH_LOSSLESS_CFG "Enable 'lossless configuration' file writing" OFF)
option(WITH_SYSTEM_BLAS "BLAS library, system installed library (with a C-style interface)" ON)
set(BLAS_ROOT " /usr/local/opt/openblas/" CACHE STRING "Path to the BLAS directory on the system")
find_package(CBLAS)

add_executable(mlp "")
set_target_properties(mlp
  PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED ON
  )
target_link_libraries(mlp PRIVATE
  mlip_cblas
  $<$<BOOL:${MPI_FOUND}>:MPI::MPI_CXX>
  $<$<CXX_COMPILER_ID:Intel>:ifcore>
  )
target_compile_definitions(mlp PRIVATE
  "MLIP_DEV"
  $<$<BOOL:${MPI_FOUND}>:"MLIP_MPI">
  $<$<NOT:$<BOOL:${WITH_SELFTEST}>>:"MLIP_NO_SELFTEST">
  $<$<BOOL:${WITH_LOSSLESS_CFG}>:"MLIP_LOSSLESS_CFG">
  $<$<BOOL:${MKL_FOUND}>:"MLIP_INTEL_MKL">
  $<$<CONFIG:Debug>:"MLIP_DEBUG">
  )

add_library(libinterface "")
set_target_properties(libinterface
  PROPERTIES
  OUTPUT_NAME "_mlip_interface"
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
  )
target_link_libraries(libinterface PRIVATE
  mlip_cblas
  $<$<CXX_COMPILER_ID:Intel>:ifcore>
  )
target_compile_definitions(libinterface PRIVATE
  "MLIP_DEV"
  $<$<BOOL:${WITH_LOSSLESS_CFG}>:"MLIP_LOSSLESS_CFG">
  $<$<BOOL:${MKL_FOUND}>:"MLIP_INTEL_MKL">
  $<$<CONFIG:Debug>:"MLIP_DEBUG">
  )

install(TARGETS mlp libinterface)

enable_testing()

add_subdirectory(src)
add_subdirectory(test)
