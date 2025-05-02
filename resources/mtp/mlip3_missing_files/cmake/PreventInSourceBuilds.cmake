# - Prevent in-source builds.
# https://stackoverflow.com/questions/1208681/with-cmake-how-would-you-disable-in-source-builds/

function(prevent_in_source_builds)

  # make sure the user doesn't play dirty with symlinks
  get_filename_component(srcdir1 "${CMAKE_SOURCE_DIR}"    REALPATH)
  get_filename_component(srcdir2 "${MLIP_SOURCE_DIR}"     REALPATH)
  get_filename_component(srcdir3 "${MLIP_SOURCE_DEV_DIR}" REALPATH)
  get_filename_component(bindir  "${CMAKE_BINARY_DIR}"    REALPATH)

  # disallow in-source builds
  if("${srcdir1}" STREQUAL "${bindir}" OR
     "${srcdir2}" STREQUAL "${bindir}" OR
     "${srcdir3}" STREQUAL "${bindir}")
    message(FATAL_ERROR "\

One must not run the CMake within the source directory. \
Instead, create a dedicated ``build`` directory and run CMake there. \
To clean up after this aborted in-place compilation: `rm -fr CMakeCache.txt CMakeFiles`
")
  endif("${srcdir1}" STREQUAL "${bindir}" OR
        "${srcdir2}" STREQUAL "${bindir}" OR
        "${srcdir3}" STREQUAL "${bindir}")
endfunction()

prevent_in_source_builds()
