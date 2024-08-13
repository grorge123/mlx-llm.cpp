cmake_minimum_required(VERSION 3.27)
project(mlx_cpp LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG e69e5f977d458f2650bb346dadf2ad30c5320281)
FetchContent_MakeAvailable(fmt)

find_package(MLX CONFIG REQUIRED)

add_library(
  mlx_cpp
  model/transformer.cpp
  model/converter.cpp
  model/utils.cpp
  mlx/base.cpp
  mlx/linear.cpp
  mlx/positional_encoding.cpp
  mlx/activations.cpp
  mlx/embedding.cpp
  mlx/normalization.cpp
  mlx/transformer.cpp)
target_link_libraries(mlx_cpp PUBLIC mlx)
target_include_directories(mlx_cpp PUBLIC ${CMAKE_SOURCE_DIR}/mlx)
target_include_directories(mlx_cpp PUBLIC ${MLX_INCLUDE_DIRS})
target_link_libraries(mlx_cpp PUBLIC ${MLX_LIBRARIES})

message(STATUS "Downloading gguflib")
FetchContent_Declare(
  gguflib
  GIT_REPOSITORY https://github.com/antirez/gguf-tools/
  GIT_TAG af7d88d808a7608a33723fba067036202910acb3)
FetchContent_MakeAvailable(gguflib)
target_include_directories(mlx_cpp
                           PRIVATE $<BUILD_INTERFACE:${gguflib_SOURCE_DIR}>)
add_library(gguflib STATIC ${gguflib_SOURCE_DIR}/fp16.c
                           ${gguflib_SOURCE_DIR}/gguflib.c)
set_target_properties(gguflib PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(mlx_cpp PRIVATE gguflib)


add_executable(main main.cpp)
target_link_libraries(main PUBLIC mlx_cpp)

add_subdirectory(tokenizer tokenizers EXCLUDE_FROM_ALL)
target_include_directories(main PUBLIC ${CMAKE_SOURCE_DIR}/tokenizer/include)
target_link_libraries(main PUBLIC tokenizers_cpp)
