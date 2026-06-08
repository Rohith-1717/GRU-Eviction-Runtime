import json
from pathlib import Path

import numpy as np


def flatten(matrix):
    return matrix.reshape(-1).tolist()


def transpose_flatten(matrix):
    return matrix.T.reshape(-1).tolist()


def write_array(handle, name, values):
    handle.write(f"static const float {name}[{len(values)}] = {{\n")

    for i, value in enumerate(values):
        suffix = "," if i + 1 < len(values) else ""
        handle.write(f"    {float(value):.8e}f{suffix}\n")

    handle.write("};\n\n")


def export_c_header(params, output_path):

    if "params" in params:
        params = params["params"]

    encoder = params["encoder"]
    scorer = params["scorer"]
    gate_kernel = encoder["gru_cell"]["gates"]["kernel"]
    candidate_kernel = encoder["gru_cell"]["candidate"]["kernel"]
    gate_bias = encoder["gru_cell"]["gates"]["bias"]
    candidate_bias = encoder["gru_cell"]["candidate"]["bias"]
    hidden_size = candidate_kernel.shape[1]
    input_size = gate_kernel.shape[0] - hidden_size
    wir = transpose_flatten(gate_kernel[:input_size, :hidden_size])
    wiz = transpose_flatten(gate_kernel[:input_size, hidden_size:2 * hidden_size])
    whr = transpose_flatten(gate_kernel[input_size:, :hidden_size])
    whz = transpose_flatten(gate_kernel[input_size:, hidden_size:2 * hidden_size])
    win = transpose_flatten(candidate_kernel[:input_size, :hidden_size])
    whn = transpose_flatten(candidate_kernel[input_size:, :hidden_size])

    bir = gate_bias[:hidden_size].tolist()
    biz = gate_bias[hidden_size:2 * hidden_size].tolist()
    bin_bias = candidate_bias.tolist()

    fc1_w = transpose_flatten(scorer["fc1"]["kernel"])
    fc1_b = scorer["fc1"]["bias"].tolist()

    fc2_w = transpose_flatten(scorer["fc2"]["kernel"])
    fc2_b = scorer["fc2"]["bias"].tolist()

    out_w = transpose_flatten(scorer["output"]["kernel"])
    out_b = scorer["output"]["bias"].tolist()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:

        handle.write("#pragma once\n")
        handle.write("#include <cstddef>\n\n")

        handle.write(f"constexpr size_t GRU_INPUT_SIZE = {input_size};\n")
        handle.write(f"constexpr size_t GRU_HIDDEN_SIZE = {hidden_size};\n")
        handle.write("constexpr size_t PAGE_FEATURE_SIZE = 4;\n")
        handle.write("constexpr size_t FC1_SIZE = 64;\n")
        handle.write("constexpr size_t FC2_SIZE = 32;\n\n")
        write_array(handle, "GRU_WIR", wir)
        write_array(handle, "GRU_WIZ", wiz)
        write_array(handle, "GRU_WIN", win)
        write_array(handle, "GRU_WHR", whr)
        write_array(handle, "GRU_WHZ", whz)
        write_array(handle, "GRU_WHN", whn)
        write_array(handle, "GRU_BIR", bir)
        write_array(handle, "GRU_BIZ", biz)
        write_array(handle, "GRU_BIN", bin_bias)
        write_array(handle, "GRU_BIR_HIDDEN", [0.0] * hidden_size)
        write_array(handle, "GRU_BIZ_HIDDEN", [0.0] * hidden_size)
        write_array(handle, "GRU_BHN", [0.0] * hidden_size)

        write_array(handle, "FC1_W", fc1_w)
        write_array(handle, "FC1_B", fc1_b)

        write_array(handle, "FC2_W", fc2_w)
        write_array(handle, "FC2_B", fc2_b)

        write_array(handle, "OUT_W", out_w)
        write_array(handle, "OUT_B", out_b)


def export_json(params, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)
