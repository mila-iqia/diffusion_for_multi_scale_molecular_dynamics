#
# The script to detect Intel(R) Math Kernel Library (MKL)
# headers and libraries.
#
# On return this will define:
#
# MKL_FOUND         - True if Intel MKL found
# MKL_INCLUDE_DIRS  - MKL include folder, where to find mkl.h, etc.
# MKL_LIBRARIES     - MKL libraries when using mkl.
#

set(MKL_INCLUDE_DIRS_HINTS
  $ENV{MKLROOT}/include
  $ENV{MKLROOT}
  $ENV{MKL_ROOT_DIR}/include
  $ENV{MKL_ROOT_DIR}
  /opt/intel/mkl/include
  ${MKL_ROOT_DIR}/include
)

set(MKL_LIBRARIES_HINTS
  $ENV{MKLROOT}/lib
  $ENV{MKLROOT}/lib/intel64
  $ENV{MKLROOT}/lib/ia32
  $ENV{MKLROOT}
  $ENV{MKL_ROOT_DIR}/lib
  $ENV{MKL_ROOT_DIR}/lib/intel64
  $ENV{MKL_ROOT_DIR}/lib/ia32
  $ENV{MKL_ROOT_DIR}
  /opt/intel/mkl/lib
  /opt/intel/mkl/lib/intel64
  /opt/intel/mkl/lib/ia32
  ${MKL_ROOT_DIR}/lib
  ${MKL_ROOT_DIR}/lib/intel64
  ${MKL_ROOT_DIR}/lib/ia32
)

find_path(MKL_INCLUDE_DIRS NAMES mkl.h HINTS ${MKL_INCLUDE_DIRS_HINTS})
find_library(MKL_LIBRARIES NAMES mkl_rt HINTS ${MKL_LIBRARIES_HINTS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL DEFAULT_MSG MKL_INCLUDE_DIRS MKL_LIBRARIES)

if(NOT MKL_FOUND)
  if(NOT DEFINED ENV{MKLROOT})
     message(WARNING "\

Set the ``MKLROOT`` environment variable to a directory that contains an MKL installation.
")
  endif(NOT DEFINED ENV{MKLROOT})
endif(NOT MKL_FOUND)

mark_as_advanced(MKL_INCLUDE_DIRS MKL_LIBRARIES MKL)
