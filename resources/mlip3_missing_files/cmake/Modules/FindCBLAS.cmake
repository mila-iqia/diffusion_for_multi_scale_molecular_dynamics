#
# The script to detect CBLAS headers and libraries
#
# The default option (WITH_SYSTEM_BLAS=ON), is to use the system installed
# BLAS library and its C interface. The priority is with the Intel
# compiler and the MKL package if it exists with its environment variables
# set. Then it searches for OpenBLAS installation, and next, it looks for
# other libraries like Atlas or LAPACK and checks if it can use them. In
# case it can not be used or is unable to find any system installed
# libraries, it would build its embedded library and use it.
# In the case of WITH_SYSTEM_BLAS=OFF value, it would escape the system
# search and build its embedded library and use it.
#
# To use the specific user/system installed library, you can set a BLAS_ROOT
# variable to a directory that contains a BLAS installation.
#
# It will use a system built CBLAS or build an
# embedded one and define:
#
# CBLAS_FOUND         - True if CBLAS found
# CBLAS_INCLUDE_DIRS  - CBLAS include folder, where to find cblas.h.
# CBLAS_LIBRARIES     - CBLAS libraries.
#

if(BLAS_ROOT)
  if(IS_DIRECTORY "${BLAS_ROOT}")
    set(CBLAS_INCLUDE_DIRS_HINTS ${BLAS_ROOT}/include ${BLAS_ROOT})
    set(CBLAS_LIBRARIES_HINTS ${BLAS_ROOT}/lib ${BLAS_ROOT}/lib64 ${BLAS_ROOT})

    find_path(CBLAS_INCLUDE_DIRS NAMES cblas.h HINTS ${CBLAS_INCLUDE_DIRS_HINTS})
    find_library(CBLAS_LIBRARIES NAMES cblas HINTS ${CBLAS_LIBRARIES_HINTS})

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(CBLAS DEFAULT_MSG CBLAS_INCLUDE_DIRS CBLAS_LIBRARIES)

    if(CBLAS_FOUND)
      add_library(mlip_cblas UNKNOWN IMPORTED)
      set_target_properties(mlip_cblas
        PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CBLAS_INCLUDE_DIRS}"
        IMPORTED_LOCATION "${CBLAS_LIBRARIES}"
      )
      unset(WITH_SYSTEM_BLAS)
    endif(CBLAS_FOUND)
  else()
    message(FATAL_ERROR "Wrong option, BLAS_ROOT = ${BLAS_ROOT} is not a directory")
  endif(IS_DIRECTORY "${BLAS_ROOT}")
endif(BLAS_ROOT)

if(WITH_SYSTEM_BLAS)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    find_package(MKL)
    if(MKL_FOUND)
      set(CBLAS_FOUND ON)
      add_library(mlip_cblas UNKNOWN IMPORTED)
      set_target_properties(mlip_cblas
        PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIRS}"
        IMPORTED_LOCATION "${MKL_LIBRARIES}"
      )
    endif(MKL_FOUND)
  endif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")

  if(NOT CBLAS_FOUND)
    find_package(OpenBLAS)
    if(OpenBLAS_FOUND)
      set(CBLAS_FOUND ON)
      add_library(mlip_cblas UNKNOWN IMPORTED)
      set_target_properties(mlip_cblas
        PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIRS}"
        IMPORTED_LOCATION "${OpenBLAS_LIBRARIES}"
      )
    endif(OpenBLAS_FOUND)
  endif(NOT CBLAS_FOUND)

  if(NOT CBLAS_FOUND)
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
      include(CheckSymbolExists)
      set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
      check_symbol_exists(cblas_dgemm "cblas.h" BLAS_HAS_CBLAS)
      list(REMOVE_ITEM CMAKE_REQUIRED_LIBRARIES "${BLAS_LIBRARIES}")

      if(BLAS_HAS_CBLAS)
        set(CBLAS_INCLUDE_DIRS_HINTS
          $ENV{Atlas_HOME}/include/atlas
          $ENV{Atlas_HOME}/include/atlas-base
          $ENV{Atlas_HOME}/include
          $ENV{Atlas_DIR}/include/atlas
          $ENV{Atlas_DIR}/include/atlas-base
          $ENV{Atlas_DIR}/include
          $ENV{Atlas_ROOT_DIR}/include/atlas
          $ENV{Atlas_ROOT_DIR}/include/atlas-base
          $ENV{Atlas_ROOT_DIR}/include
          $ENV{CBLASDIR}/include
          $ENV{CBLASDIR}
          /usr/local/opt/atlas/include
          /usr/local/include/atlas
          /usr/local/include/atlas-base
          /usr/local/include
          /usr/include/atlas
          /usr/include/atlas-base
          /usr/include
        )

        set(CBLAS_LIBRARIES_HINTS
          $ENV{Atlas_HOME}/lib
          $ENV{Atlas_HOME}
          $ENV{Atlas_DIR}/lib
          $ENV{Atlas_DIR}
          $ENV{Atlas_ROOT_DIR}/lib
          $ENV{Atlas_ROOT_DIR}
          $ENV{CBLASDIR}/lib
          $ENV{CBLASDIR}/lib64
          /usr/local/opt/atlas/lib
          /usr/lib/atlas
          /usr/lib/atlas-base
          /usr/lib
          /usr/lib64/atlas
          /usr/lib64/atlas-base
          /usr/lib64
          /usr/local/lib64/atlas
          /usr/local/lib64/atlas-base
          /usr/local/lib64
          /usr/local/lib/atlas
          /usr/local/lib/atlas-base
          /usr/local/lib
          /lib/atlas
          /lib/atlas-base
          /lib
          /lib64/atlas
          /lib64/atlas-base
          /lib64
        )

        foreach(src_file ${BLAS_LIBRARIES})
          get_filename_component(src_file_path "${src_file}" DIRECTORY)
          list(APPEND ${CBLAS_INCLUDE_DIRS_HINTS} "${src_file_path}")
          list(APPEND ${CBLAS_LIBRARIES_HINTS} "${src_file_path}")
        endforeach(src_file ${BLAS_LIBRARIES})

        find_path(CBLAS_INCLUDE_DIRS NAMES cblas.h HINTS ${CBLAS_INCLUDE_DIRS_HINTS})
        find_library(CBLAS_LIBRARIES NAMES cblas HINTS ${CBLAS_LIBRARIES_HINTS})

        include(FindPackageHandleStandardArgs)
        find_package_handle_standard_args(CBLAS DEFAULT_MSG CBLAS_INCLUDE_DIRS CBLAS_LIBRARIES)

        if(CBLAS_FOUND)
          add_library(mlip_cblas UNKNOWN IMPORTED)
          set_target_properties(mlip_cblas
            PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CBLAS_INCLUDE_DIRS}"
            IMPORTED_LOCATION "${CBLAS_LIBRARIES}"
          )
        endif(CBLAS_FOUND)
      endif(BLAS_HAS_CBLAS)
    endif(BLAS_FOUND)
  endif(NOT CBLAS_FOUND)

  if(NOT CBLAS_FOUND)
    if(NOT BLAS_ROOT)
      message(WARNING " \

To use the specific system installed library you can set \
a ``BLAS_ROOT`` variable to a directory that contains a CBLAS \
installation: `cmake .. -DBLAS_ROOT=<Path to a directory on disk>`")
    endif(NOT BLAS_ROOT)
  endif(NOT CBLAS_FOUND)
endif(WITH_SYSTEM_BLAS)

if(NOT CBLAS_FOUND)
  add_subdirectory(${CMAKE_SOURCE_DIR}/cblas)
endif(NOT CBLAS_FOUND)

mark_as_advanced(CBLAS_INCLUDE_DIRS CBLAS_LIBRARIES CBLAS)
