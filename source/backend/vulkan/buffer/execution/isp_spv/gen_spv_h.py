import os
for shader in ['isp_display_uint32_rgba', 'isp_display_uint32_bgra', 'isp_display_uint32_argb']:
    with open(shader + '.spv', 'rb') as f:
        data = f.read()
    with open(shader + '_spv.h', 'wb') as f:
        f.write(b'const unsigned char g_' + shader.encode() + b'_spv[] = {\n')
        for i in range(0, len(data), 16):
            chunk = data[i:i+16]
            hex_str = ', '.join('0x{:02x}'.format(b) for b in chunk)
            f.write(('  ' + hex_str + ',\n').encode())
        f.write(b'};\n')
        f.write(('const int g_' + shader + '_spv_len = ' + str(len(data)) + ';\n').encode())
    print(f'Generated {shader}_spv.h, size: {len(data)}')

# Generate SPIR-V header files for minimal opset shaders
for shader in ['isp_lsc_min', 'isp_vignetting_min', 'isp_chromatic_aberration_min', 'isp_raw_blc_min']:
    spv_file = shader + '.spv'
    h_file = shader + '_spv.h'
    if not os.path.exists(spv_file):
        print(f'ERROR: {spv_file} not found')
        continue
    with open(spv_file, 'rb') as f:
        data = f.read()
    with open(h_file, 'wb') as f:
        f.write(b'const unsigned char g_' + shader.encode() + b'_spv[] = {\n')
        for i in range(0, len(data), 16):
            chunk = data[i:i+16]
            hex_str = ', '.join('0x{:02x}'.format(b) for b in chunk)
            f.write(('  ' + hex_str + ',\n').encode())
        f.write(b'};\n')
        f.write(('const int g_' + shader + '_spv_len = ' + str(len(data)) + ';\n').encode())
    print(f'Generated {h_file}, size: {len(data)}')
