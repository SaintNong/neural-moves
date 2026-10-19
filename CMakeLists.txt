cmake_minimum_required(VERSION 3.20)
project(neural_move_ordering LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
endif()

include(CTest)
option(NMO_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
set(NMO_RUN_DEPTH 3 CACHE STRING "Search depth for the run target")

add_library(nmo_build_options INTERFACE)
target_compile_features(nmo_build_options INTERFACE cxx_std_20)
if(MSVC)
    target_compile_options(nmo_build_options INTERFACE /W4 /permissive-)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(nmo_build_options INTERFACE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()
if(NMO_SANITIZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" OR MSVC)
        message(FATAL_ERROR "NMO_SANITIZERS requires GCC or Clang with GNU-style flags")
    endif()
    target_compile_options(nmo_build_options INTERFACE
        -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(nmo_build_options INTERFACE -fsanitize=address,undefined)
endif()

add_library(chess_library INTERFACE)
target_include_directories(chess_library SYSTEM INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/chess-library")

add_library(nmo STATIC
    src/evaluation.cpp
    src/ordering.cpp
    src/positions.cpp
    src/search.cpp
    src/transposition_table.cpp)
target_include_directories(nmo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(nmo PUBLIC chess_library nmo_build_options)

add_executable(neural-move-ordering src/main.cpp)
target_link_libraries(neural-move-ordering PRIVATE nmo)

add_custom_target(run
    COMMAND $<TARGET_FILE:neural-move-ordering>
        "${CMAKE_CURRENT_SOURCE_DIR}/positions/quiet-labeled-5000.fen"
        --depth "${NMO_RUN_DEPTH}"
    DEPENDS neural-move-ordering
    USES_TERMINAL
    VERBATIM)

if(BUILD_TESTING)
    add_executable(nmo-tests
        tests/main.cpp
        tests/evaluation_test.cpp
        tests/ordering_test.cpp
        tests/positions_test.cpp
        tests/search_test.cpp)
    target_include_directories(nmo-tests SYSTEM PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/doctest")
    target_link_libraries(nmo-tests PRIVATE nmo)
    add_test(NAME nmo-tests COMMAND nmo-tests)
    set_tests_properties(nmo-tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
