#include <cstdio>
#include <cmath>
#include <cstdint>

// ExecuTorch headers FIRST (before Pico SDK to avoid macro conflicts)
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>

// Pico SDK headers AFTER ExecuTorch
#include "pico/stdlib.h"
#include "pico/cyw43_arch.h"

#include "model_pte.h"

using namespace executorch::runtime;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::aten::ScalarType;

static uint8_t method_allocator_pool[32 * 1024];
static uint8_t activation_pool[16 * 1024];

void blink_led(int times, int delay_ms = 200) {
    for (int i = 0; i < times; i++) {
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1);
        sleep_ms(delay_ms);
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 0);
        sleep_ms(delay_ms);
    }
}

void wait_for_usb() {
    int retries = 0;
    while (!stdio_usb_connected() && retries < 50) {
        sleep_ms(100);
        retries++;
    }
    sleep_ms(500);
}

float run_inference(Method& method, float input_value) {
    float input_data[1] = {input_value};
    TensorImpl::SizesType input_sizes[2] = {1, 1};
    TensorImpl::DimOrderType dim_order[2] = {0, 1};

    TensorImpl input_impl(ScalarType::Float, 2, input_sizes, input_data, dim_order);
    Tensor input_tensor(&input_impl);

    if (method.set_input(input_tensor, 0) != Error::Ok) {
        printf("ERROR: set_input failed\n");
        return 0.0f;
    }

    if (method.execute() != Error::Ok) {
        printf("ERROR: execute failed\n");
        return 0.0f;
    }

    auto output = method.get_output(0);
    if (!output.isTensor()) {
        printf("ERROR: output not tensor\n");
        return 0.0f;
    }

    return output.toTensor().const_data_ptr<float>()[0];
}

int main() {
    stdio_init_all();
    
    if (cyw43_arch_init()) {
        printf("ERROR: CYW43 init failed\n");
        return 1;
    }

    wait_for_usb();
    runtime_init();

    printf("========================================\n");
    printf("  ExecuTorch Sine Wave Predictor\n");
    printf("  Raspberry Pi Pico 2 W\n");
    printf("========================================\n\n");

    MemoryAllocator method_allocator(sizeof(method_allocator_pool), method_allocator_pool);
    Span<uint8_t> planned_buffers[1] = {{activation_pool, sizeof(activation_pool)}};
    HierarchicalAllocator planned_memory({planned_buffers, 1});
    MemoryManager memory_manager(&method_allocator, &planned_memory);

    printf("Loading model (%u bytes)...\n", model_pte_len);
    executorch::extension::BufferDataLoader loader(model_pte, model_pte_len);

    auto program_result = Program::load(&loader);
    if (!program_result.ok()) {
        printf("ERROR: Program load failed (%d)\n", (int)program_result.error());
        blink_led(10, 100);
        while (1) tight_loop_contents();
    }

    Program program = std::move(*program_result);
    auto method_name = program.get_method_name(0);
    if (!method_name.ok()) {
        printf("ERROR: get_method_name failed\n");
        blink_led(10, 100);
        while (1) tight_loop_contents();
    }

    printf("Loading method '%s'...\n", *method_name);
    auto method_result = program.load_method(*method_name, &memory_manager);
    if (!method_result.ok()) {
        printf("ERROR: load_method failed (%d)\n", (int)method_result.error());
        blink_led(10, 100);
        while (1) tight_loop_contents();
    }

    Method method = std::move(*method_result);
    printf("Model ready!\n\n");
    blink_led(3, 300);

    const float PI = 3.14159265358979f;
    const int NUM_POINTS = 100;
    const float step = (2.0f * PI) / NUM_POINTS;

    printf("Starting inference...\nFormat: DATA,x,predicted,expected\n\n");

    while (true) {
        for (int i = 0; i <= NUM_POINTS; i++) {
            float x = i * step;
            float predicted = run_inference(method, x);
            float expected = sinf(x);
            printf("DATA,%.4f,%.4f,%.4f\n", x, predicted, expected);
            cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, i % 10 == 0);
            sleep_ms(50);
        }
        printf("--- Cycle complete ---\n");
        sleep_ms(1000);
    }

    return 0;
}