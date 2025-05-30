 /*
 * Copyright (c) 2023-2025 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_HEXAGON_MAX_DEVICES    3
#define GGML_HEXAGON_BACKEND_NAME   "hexagon"

enum HEXAGONBackend {
    HEXAGON_BACKEND_QNNCPU  = 0,
    HEXAGON_BACKEND_QNNGPU  = 1,
    HEXAGON_BACKEND_QNNNPU  = 2,
    HEXAGON_BACKEND_CDSP    = 2,
    HEXAGON_BACKEND_GGML    = 3, //"fake" QNN backend for compare performance between HEXAGON backend and ggml backend
};

GGML_BACKEND_API ggml_backend_t ggml_backend_hexagon_init(size_t dev_num, const char * qnn_lib_path);

GGML_BACKEND_API bool           ggml_backend_is_hexagon(ggml_backend_t backend);

GGML_BACKEND_API int            ggml_backend_hexagon_get_device_count(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_hexagon_reg(void);

const char * ggml_backend_hexagon_get_devname(size_t dev_num);

#ifdef __cplusplus
}
#endif
