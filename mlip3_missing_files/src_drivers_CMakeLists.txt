set(SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/basic_drivers.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/relaxation.cpp
  )

target_sources(mlp PRIVATE ${SOURCES})
target_sources(libinterface PRIVATE ${SOURCES})
