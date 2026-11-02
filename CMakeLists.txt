cmake_minimum_required(VERSION 3.14)
project(energy_based_isotropic_remeshing LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build dependencies as static libraries" FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(cmake/CPM.cmake)

CPMAddPackage(
    NAME pmp-library
    GITHUB_REPOSITORY pmp-library/pmp-library
    GIT_TAG 3.0.0
    OPTIONS
        "PMP_BUILD_VIS OFF"
        "PMP_BUILD_APPS OFF"
        "PMP_BUILD_EXAMPLES OFF"
        "PMP_BUILD_TESTS OFF"
)

CPMAddPackage(
    NAME polyscope
    GITHUB_REPOSITORY nmwsharp/polyscope
    GIT_TAG v2.6.1
)

CPMAddPackage(
    NAME portable_file_dialogs
    GITHUB_REPOSITORY samhocevar/portable-file-dialogs
    GIT_TAG 0.1.0
)

add_library(remeshing STATIC
    src/remeshing/batched.cpp
    src/remeshing/combined.cpp
    src/remeshing/cycles.cpp
    src/remeshing/dynamic_queue.cpp
    src/remeshing/edits.cpp
    src/remeshing/evaluate.cpp
    src/remeshing/metrics.cpp
    src/remeshing/remesh.cpp
    src/remeshing/validity.cpp
)
target_include_directories(remeshing PUBLIC src)
target_link_libraries(remeshing PUBLIC pmp)

add_executable(remesher-cli apps/remesher-cli/main.cpp)
target_include_directories(remesher-cli PRIVATE
    "${polyscope_SOURCE_DIR}/deps/args"
    "${polyscope_SOURCE_DIR}/deps/json/include"
)
target_link_libraries(remesher-cli PRIVATE remeshing)

# Loss colouring and camera handling of the three programs with a window.
add_library(visuals STATIC apps/shared/visuals.cpp)
target_include_directories(visuals PUBLIC apps)
target_link_libraries(visuals PUBLIC remeshing polyscope)

add_executable(remesher apps/remesher/main.cpp)
target_compile_definitions(remesher PRIVATE MESHES_DIR="${CMAKE_SOURCE_DIR}/meshes")
target_link_libraries(remesher PRIVATE visuals portable_file_dialogs)

add_executable(mesh-screenshot apps/mesh-screenshot/main.cpp)
target_include_directories(mesh-screenshot PRIVATE
    "${polyscope_SOURCE_DIR}/deps/args"
    "${polyscope_SOURCE_DIR}/deps/json/include"
)
target_link_libraries(mesh-screenshot PRIVATE visuals)

add_executable(viewer apps/viewer/main.cpp)
target_include_directories(viewer PRIVATE "${polyscope_SOURCE_DIR}/deps/json/include")
target_link_libraries(viewer PRIVATE visuals)

# mesh_preparation/prepare_meshes.py: not part of the default build.
add_executable(fill-holes EXCLUDE_FROM_ALL mesh_preparation/fill_holes.cpp)
target_link_libraries(fill-holes PRIVATE pmp)
