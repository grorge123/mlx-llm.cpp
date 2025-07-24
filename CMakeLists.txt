cmake_minimum_required(VERSION 3.27)
project(mlx_cpp LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined
# -fsanitize=address -g")

include(FetchContent)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG e69e5f977d458f2650bb346dadf2ad30c5320281)
FetchContent_MakeAvailable(fmt)
find_package(MLX CONFIG REQUIRED)
find_package(ZLIB REQUIRED)

find_program(FFMPEG_EXECUTABLE ffmpeg)
if(NOT FFMPEG_EXECUTABLE)
  message(WARNING "ffmpeg not found. Audio loading functionality in whisper_transcribe.cpp requires ffmpeg CLI tool to be installed and available in PATH.")
else()
  message(STATUS "Found ffmpeg: ${FFMPEG_EXECUTABLE}")
endif()
add_library(
  mlx_cpp
  prompt/prompt.cpp
  model/llm/transformer.cpp
  model/llm/registry.cpp
  model/mllama/language.cpp
  model/mllama/vision.cpp
  model/mllama/mllama.cpp
  model/gemma3/language.cpp
  model/gemma3/vision.cpp
  model/gemma3/gemma3.cpp
  model/converter.cpp
  model/utils.cpp
  model/vlm_base.cpp
  model/vlm_sampling.cpp
  model/whisper/whisper.cpp
  model/whisper/tokenizer.cpp
  model/whisper/decoding.cpp
  model/whisper_transcribe.cpp
  mlx/base.cpp
  mlx/linear.cpp
  mlx/convolution.cpp
  mlx/positional_encoding.cpp
  mlx/activations.cpp
  mlx/embedding.cpp
  mlx/normalization.cpp
  mlx/transformer.cpp
  mlx/pooling.cpp
  mlx/quantized.cpp)
target_compile_options(mlx_cpp PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(mlx_cpp PUBLIC mlx)
target_include_directories(mlx_cpp PUBLIC ${CMAKE_SOURCE_DIR}/mlx)
target_include_directories(mlx_cpp PUBLIC ${MLX_INCLUDE_DIRS})
target_link_libraries(mlx_cpp PUBLIC ${MLX_LIBRARIES})
target_link_libraries(mlx_cpp PUBLIC ZLIB::ZLIB)

target_compile_options(
  mlx_cpp
  PRIVATE -Wall
          -Wextra
          -Wpedantic
          -Werror
          -Wnounused-const-variable
          -fsanitize=undefined
          -fsanitize=address
          -g)

target_link_options(mlx_cpp PRIVATE -fsanitize=undefined -fsanitize=address)

target_link_libraries(mlx_cpp PRIVATE "-fsanitize=address"
                                      "-fsanitize=undefined"
                                      "-g")

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
  GIT_TAG v1.15.3)
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

target_include_directories(mlx_cpp PUBLIC ${CMAKE_SOURCE_DIR}/tokenizer/include)
target_link_libraries(mlx_cpp PUBLIC tokenizers_cpp)
target_include_directories(llm PUBLIC ${CMAKE_SOURCE_DIR}/tokenizer/include)
target_link_libraries(llm PUBLIC tokenizers_cpp)

add_executable(whisper example/whisper.cpp)
target_include_directories(whisper PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(whisper PUBLIC mlx_cpp)
target_compile_options(whisper PRIVATE -Wall -Wextra -Wpedantic -Werror)
