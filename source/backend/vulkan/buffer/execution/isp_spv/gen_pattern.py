import os
shader = 'isp_pattern'
spv = shader + '.spv'
h = shader + '_spv.h'
with open(spv, 'rb') as f:
    data = f.read()
with open(h, 'w') as f:
    f.write('const unsigned char g_%s_spv[] = {\n' % shader)
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        hex_str = ', '.join('0x{:02x}'.format(b) for b in chunk)
        f.write('  %s,\n' % hex_str)
    f.write('};\n')
    f.write('const int g_%s_spv_len = %d;\n' % (shader, len(data)))
print('Generated %s, %d bytes' % (h, len(data)))
