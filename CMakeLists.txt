cmake_minimum_required(VERSION 3.13)

set(PICO_BOARD pico2_w)

include($ENV{PICO_SDK_PATH}/pico_sdk_init.cmake)

project(sine_predictor C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

if(NOT DEFINED EXECUTORCH_ROOT)
    message(FATAL_ERROR "EXECUTORCH_ROOT not set")
endif()

set(ET_BUILD_DIR ${EXECUTORCH_ROOT}/cmake-out)

add_executable(sine_predictor main.cpp)

target_include_directories(sine_predictor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EXECUTORCH_ROOT}
    ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)

target_compile_definitions(sine_predictor PRIVATE
    C10_USING_CUSTOM_GENERATED_MACROS
    EXECUTORCH_ENABLE_LOGGING=OFF
    EXECUTORCH_PAL_DEFAULT=minimal
)

target_compile_options(sine_predictor PRIVATE
    -mcpu=cortex-m33 -mthumb -mfloat-abi=soft
    -Os -ffunction-sections -fdata-sections
)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")

target_link_libraries(sine_predictor PRIVATE
    ${ET_BUILD_DIR}/lib/libexecutorch.a
    ${ET_BUILD_DIR}/lib/libexecutorch_core.a
    -Wl,--whole-archive
    ${ET_BUILD_DIR}/lib/libportable_ops_lib.a
    -Wl,--no-whole-archive
    ${ET_BUILD_DIR}/lib/libportable_kernels.a
    pico_stdlib
    pico_cyw43_arch_none
)

pico_enable_stdio_usb(sine_predictor 1)
pico_enable_stdio_uart(sine_predictor 0)
pico_add_extra_outputs(sine_predictor)