#
# The script to detect OpenBLAS Library
# headers and libraries.
#
# On return this will define:
#
# OpenBLAS_FOUND         - True if OpenBLAS found
# OpenBLAS_INCLUDE_DIRS  - OpenBLAS include folder, where to find cblas.h.
# OpenBLAS_LIBRARIES     - OpenBLAS libraries.
#

set(OpenBLAS_INCLUDE_DIRS_HINTS
  $ENV{OpenBLAS}/include/openblas
  $ENV{OpenBLAS}/include
  $ENV{OpenBLAS}
  $ENV{OpenBLAS_HOME}/include/openblas
  $ENV{OpenBLAS_HOME}/include
  $ENV{OpenBLAS_HOME}
  $ENV{OpenBLAS_DIR}/include/openblas
  $ENV{OpenBLAS_DIR}/include
  $ENV{OpenBLAS_DIR}
  /usr/local/opt/openblas/include
  /opt/OpenBLAS/include
  /usr/local/include/openblas
  /usr/include/openblas
  /usr/local/include/openblas-base
  /usr/include/openblas-base
  /usr/local/include
  /usr/include
)

set(OpenBLAS_LIBRARIES_HINTS
  $ENV{OpenBLAS}/lib
  $ENV{OpenBLAS}
  $ENV{OpenBLAS_HOME}/lib
  $ENV{OpenBLAS_HOME}
  $ENV{OpenBLAS_DIR}/lib
  $ENV{OpenBLAS_DIR}
  /usr/local/opt/openblas/lib
  /opt/OpenBLAS/lib
  /usr/local/lib64
  /usr/local/lib
  /lib/openblas-base
  /lib64
  /lib
  /usr/lib/openblas-base
  /usr/lib64
  /usr/lib
)

find_path(OpenBLAS_INCLUDE_DIRS NAMES cblas.h HINTS ${OpenBLAS_INCLUDE_DIRS_HINTS})
find_library(OpenBLAS_LIBRARIES NAMES openblas HINTS ${OpenBLAS_LIBRARIES_HINTS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenBLAS DEFAULT_MSG OpenBLAS_INCLUDE_DIRS OpenBLAS_LIBRARIES)

mark_as_advanced(OpenBLAS_INCLUDE_DIRS OpenBLAS_LIBRARIES OpenBLAS)
