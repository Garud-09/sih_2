"""
Converts a binary .tflite model file into a C++ header file containing
an aligned byte array ready for TensorFlow Lite for Microcontrollers (TFLM).
"""

import os
import sys


def convert_to_c_header(tflite_path, output_header_path, array_name="g_model"):
    """Reads binary TFLite file and writes out a C++ header file."""
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"TFLite file not found at: {tflite_path}")

    with open(tflite_path, "rb") as f:
        bytes_data = f.read()

    total_bytes = len(bytes_data)

    lines = []
    lines.append("// Auto-generated TensorFlow Lite Micro Model Header")
    lines.append("// Target: ESP32-S3 (Xtensa Dual-Core) - ISRO PS 26172")
    lines.append(f"// Total Model Size: {total_bytes} bytes ({total_bytes / 1024.0:.2f} KB)")
    lines.append("#ifndef MODEL_DATA_H_")
    lines.append("#define MODEL_DATA_H_\n")
    lines.append("#include <cstdint>\n")
    lines.append(f"// Aligned to 16 bytes for Xtensa SIMD / TFLite Micro requirement")
    lines.append(f"alignas(16) const unsigned char {array_name}[] = {{")

    # Format bytes in hex, 12 per line
    bytes_per_line = 12
    for i in range(0, total_bytes, bytes_per_line):
        chunk = bytes_data[i:i + bytes_per_line]
        hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
        if i + bytes_per_line < total_bytes:
            lines.append(f"    {hex_str},")
        else:
            lines.append(f"    {hex_str}")

    lines.append("};\n")
    lines.append(f"const int {array_name}_len = {total_bytes};\n")
    lines.append("#endif  // MODEL_DATA_H_\n")

    content = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(output_header_path)), exist_ok=True)
    with open(output_header_path, "w") as f:
        f.write(content)

    print(f"Exported {total_bytes} bytes to {output_header_path}")
    return total_bytes


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 export_tflite.py <input.tflite> <output_header.h>")
        sys.exit(1)
    convert_to_c_header(sys.argv[1], sys.argv[2])
