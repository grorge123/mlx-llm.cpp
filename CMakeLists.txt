cmake_minimum_required(VERSION 3.27)
project(mlx_cpp LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG e69e5f977d458f2650bb346dadf2ad30c5320281)
FetchContent_MakeAvailable(fmt)

find_package(MLX CONFIG REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined
# -fsanitize=address -g")
add_library(
  mlx_cpp
  prompt/prompt.cpp
  model/llm/transformer.cpp
  model/llm/registry.cpp
  model/mllama/language.cpp
  model/mllama/vision.cpp
  model/mllama/mllama.cpp
  model/converter.cpp
  model/utils.cpp
  model/vlm_base.cpp
  mlx/base.cpp
  mlx/linear.cpp
  mlx/convolution.cpp
  mlx/positional_encoding.cpp
  mlx/activations.cpp
  mlx/embedding.cpp
  mlx/normalization.cpp
  mlx/transformer.cpp
  mlx/quantized.cpp)
target_compile_options(mlx_cpp PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(mlx_cpp PUBLIC mlx)
target_include_directories(mlx_cpp PUBLIC ${CMAKE_SOURCE_DIR}/mlx)
target_include_directories(mlx_cpp PUBLIC ${MLX_INCLUDE_DIRS})
target_link_libraries(mlx_cpp PUBLIC ${MLX_LIBRARIES})

find_package(simdjson QUIET)
FetchContent_Declare(
  simdjson
  GIT_REPOSITORY https://github.com/simdjson/simdjson.git
  GIT_TAG tags/v3.10.0
  GIT_SHALLOW TRUE)
set(SIMDJSON_DEVELOPER_MODE
    OFF
    CACHE BOOL "SIMDJSON developer mode" FORCE)
FetchContent_MakeAvailable(simdjson)
set_property(TARGET simdjson PROPERTY POSITION_INDEPENDENT_CODE ON)
message(STATUS "Downloading SIMDJSON source -- done")
target_link_libraries(mlx_cpp PUBLIC simdjson::simdjson)

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

find_package(spdlog NO_CMAKE_PACKAGE_REGISTRY QUIET)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.3.1)
FetchContent_MakeAvailable(spdlog)
target_compile_options(spdlog INTERFACE -w)
target_link_libraries(mlx_cpp PUBLIC spdlog)
target_include_directories(mlx_cpp PUBLIC ${spdlog_INCLUDE_DIRS})

add_executable(vlm example/vlm.cpp)
target_include_directories(vlm PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(vlm PUBLIC mlx_cpp)
target_compile_options(vlm PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(llm example/llm.cpp)
target_include_directories(llm PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(llm PUBLIC mlx_cpp)
target_compile_options(llm PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_subdirectory(tokenizer tokenizers EXCLUDE_FROM_ALL)
target_include_directories(llm PUBLIC ${CMAKE_SOURCE_DIR}/tokenizer/include)
target_link_libraries(llm PUBLIC tokenizers_cpp)
