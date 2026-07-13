/**
 * CPU Image Process Stubs — CV function stubs for MNN_CPU=OFF builds.
 *
 * When MNN_CPU=OFF, the CPU compute directory (with functions like
 * MNNC1blitH, MNNRGBToBGR, etc.) is excluded. These stubs satisfy
 * the linker for the MNNCV module's dependency on those symbols.
 *
 * When MNN_MINIMAL_CPU=ON (CPU=ON but compute/ excluded), these stubs
 * are still needed for the same reason.
 */

#include <cstddef>
#include <cstdint>

namespace MNN {
namespace CV {
struct Point { int x; int y; };
}
} // namespace MNN

extern "C" {
void MNNMatrixAddCommon(float*, const float*, const float*, size_t, size_t) {}
void MNNMatrixSubCommon(float*, const float*, const float*, size_t, size_t) {}
void MNNMatrixProdCommon(float*, const float*, const float*, size_t, size_t) {}
}

// ── Image format conversions ──────────────────────────────────────────

void MNNC1blitH(const unsigned char* s, unsigned char* d, size_t c) {
    for (size_t i = 0; i < c; i++) d[i] = s[i];
}
void MNNC3blitH(const unsigned char* s, unsigned char* d, size_t c) {
    for (size_t i = 0; i < c * 3; i++) d[i] = s[i];
}
void MNNC4blitH(const unsigned char* s, unsigned char* d, size_t c) {
    for (size_t i = 0; i < c * 4; i++) d[i] = s[i];
}
void MNNCopyC3(const unsigned char* s, unsigned char* d, size_t c) {
    for (size_t i = 0; i < c * 3; i++) d[i] = s[i];
}
void MNNCopyC4(const unsigned char* s, unsigned char* d, size_t c) {
    for (size_t i = 0; i < c * 4; i++) d[i] = s[i];
}
void MNNGRAYToC3(const unsigned char* s, unsigned char* d, size_t c) {
    for (size_t i = 0; i < c; i++) d[i*3+0] = d[i*3+1] = d[i*3+2] = s[i];
}
void MNNGRAYToC4(const unsigned char* s, unsigned char* d, size_t c) {
    for (size_t i = 0; i < c; i++) {
        d[i*4+0] = d[i*4+1] = d[i*4+2] = s[i];
        d[i*4+3] = 255;
    }
}
void MNNRGBToBGR(const unsigned char* s, unsigned char* d, size_t c) {
    for (size_t i = 0; i < c; i++) {
        d[i*3+0] = s[i*3+2]; d[i*3+1] = s[i*3+1]; d[i*3+2] = s[i*3+0];
    }
}
void MNNRGBToGRAY(const unsigned char* s, unsigned char* d, size_t c) {
    for (size_t i = 0; i < c; i++)
        d[i] = (unsigned char)(0.299f*s[i*3+0] + 0.587f*s[i*3+1] + 0.114f*s[i*3+2]);
}
void MNNBRGToGRAY(const unsigned char* s, unsigned char* d, size_t c) {
    for (size_t i = 0; i < c; i++)
        d[i] = (unsigned char)(0.299f*s[i*3+2] + 0.587f*s[i*3+1] + 0.114f*s[i*3+0]);
}
void MNNBGRAToBGR(const unsigned char* s, unsigned char* d, size_t c) {
    for (size_t i = 0; i < c; i++) {
        d[i*3+0] = s[i*4+0]; d[i*3+1] = s[i*4+1]; d[i*3+2] = s[i*4+2];
    }
}
void MNNBGRAToGRAY(const unsigned char* s, unsigned char* d, size_t c) {
    for (size_t i = 0; i < c; i++)
        d[i] = (unsigned char)(0.299f*s[i*4+2] + 0.587f*s[i*4+1] + 0.114f*s[i*4+0]);
}
void MNNRGBAToBGR(const unsigned char* s, unsigned char* d, size_t c) {
    for (size_t i = 0; i < c; i++) {
        d[i*3+0] = s[i*4+0]; d[i*3+1] = s[i*4+1]; d[i*3+2] = s[i*4+2];
    }
}
void MNNRGBAToBGRA(const unsigned char* s, unsigned char* d, size_t c) {
    for (size_t i = 0; i < c; i++) {
        d[i*4+0] = s[i*4+2]; d[i*4+1] = s[i*4+1];
        d[i*4+2] = s[i*4+0]; d[i*4+3] = s[i*4+3];
    }
}
void MNNRGBAToGRAY(const unsigned char* s, unsigned char* d, size_t c) {
    for (size_t i = 0; i < c; i++)
        d[i] = (unsigned char)(0.299f*s[i*4+0] + 0.587f*s[i*4+1] + 0.114f*s[i*4+2]);
}
void MNNC3ToBGR555(const unsigned char*, unsigned char*, size_t, bool) {}
void MNNC3ToBGR565(const unsigned char*, unsigned char*, size_t, bool) {}
void MNNNV21ToRGB(const unsigned char*, unsigned char*, size_t) {}
void MNNNV21ToBGR(const unsigned char*, unsigned char*, size_t) {}
void MNNNV21ToRGBA(const unsigned char*, unsigned char*, size_t) {}
void MNNNV21ToBGRA(const unsigned char*, unsigned char*, size_t) {}
void MNNC3ToHSV(const unsigned char*, unsigned char*, size_t, bool, bool) {}
void MNNC3ToXYZ(const unsigned char*, unsigned char*, size_t, bool) {}
void MNNC3ToYUV(const unsigned char*, unsigned char*, size_t, bool, bool) {}
void MNNC1ToFloatRGBA(const unsigned char*, float*, const float*, const float*, size_t) {}
void MNNC3ToFloatRGBA(const unsigned char*, float*, const float*, const float*, size_t) {}
void MNNC1ToFloatC1(const unsigned char*, float*, const float*, const float*, size_t) {}
void MNNC3ToFloatC3(const unsigned char*, float*, const float*, const float*, size_t) {}
void MNNC4ToFloatC4(const unsigned char*, float*, const float*, const float*, size_t) {}
void MNNC3ToC4(const unsigned char* s, unsigned char* d, size_t c) {
    for (size_t i = 0; i < c; i++) {
        d[i*4+0] = s[i*3+0]; d[i*4+1] = s[i*3+1];
        d[i*4+2] = s[i*3+2]; d[i*4+3] = 255;
    }
}

// ── Samplers ──────────────────────────────────────────────────────────

void MNNSamplerC1Bilinear(const unsigned char* src, unsigned char* dst,
                           MNN::CV::Point* pts, size_t sta, size_t count,
                           size_t cap, size_t iw, size_t ih, size_t stride) {}
void MNNSamplerC3Bilinear(const unsigned char* src, unsigned char* dst,
                           MNN::CV::Point* pts, size_t sta, size_t count,
                           size_t cap, size_t iw, size_t ih, size_t stride) {}
void MNNSamplerC4Bilinear(const unsigned char* src, unsigned char* dst,
                           MNN::CV::Point* pts, size_t sta, size_t count,
                           size_t cap, size_t iw, size_t ih, size_t stride) {}
void MNNSamplerC1Nearest(const unsigned char* src, unsigned char* dst,
                          MNN::CV::Point* pts, size_t sta, size_t count,
                          size_t cap, size_t iw, size_t ih, size_t stride) {}
void MNNSamplerC3Nearest(const unsigned char* src, unsigned char* dst,
                          MNN::CV::Point* pts, size_t sta, size_t count,
                          size_t cap, size_t iw, size_t ih, size_t stride) {}
void MNNSamplerC4Nearest(const unsigned char* src, unsigned char* dst,
                          MNN::CV::Point* pts, size_t sta, size_t count,
                          size_t cap, size_t iw, size_t ih, size_t stride) {}
void MNNSamplerCopyCommon(const unsigned char* src, unsigned char* dst,
                           MNN::CV::Point* pts, size_t sta, size_t count,
                           size_t iw, size_t ih, size_t stride, int bpp) {}
void MNNSamplerI420Copy(const unsigned char* src, unsigned char* dst,
                         MNN::CV::Point* pts, size_t sta, size_t count,
                         size_t cap, size_t iw, size_t ih, size_t stride) {}
void MNNSamplerNV21Copy(const unsigned char* src, unsigned char* dst,
                         MNN::CV::Point* pts, size_t sta, size_t count,
                         size_t cap, size_t iw, size_t ih, size_t stride) {}
void MNNSamplerNV12Copy(const unsigned char* src, unsigned char* dst,
                         MNN::CV::Point* pts, size_t sta, size_t count,
                         size_t cap, size_t iw, size_t ih, size_t stride) {}
void MNNSamplerI420Nearest(const unsigned char* src, unsigned char* dst,
                            MNN::CV::Point* pts, size_t sta, size_t count,
                            size_t cap, size_t iw, size_t ih, size_t stride) {}
void MNNSamplerNV21Nearest(const unsigned char* src, unsigned char* dst,
                            MNN::CV::Point* pts, size_t sta, size_t count,
                            size_t cap, size_t iw, size_t ih, size_t stride) {}
void MNNSamplerNV12Nearest(const unsigned char* src, unsigned char* dst,
                            MNN::CV::Point* pts, size_t sta, size_t count,
                            size_t cap, size_t iw, size_t ih, size_t stride) {}
