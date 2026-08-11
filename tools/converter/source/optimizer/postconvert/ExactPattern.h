// ExactPattern.h — ISP exact-match pattern struct and constructors.
//
// Extracted from IspChainFusion.cpp so the auto-generated IspExactPatterns.h
// can include it without duplicating the struct definition.
//
// Seven constructor overloads cover all table entry forms:
//   Ctor1 (base):         (nk, cwe, cwv, cv)
//   Ctor2 (binOp):        (bot, nk, cwe, cwv, cv)
//   Ctor3 (guard):        (bool nf, ccc)
//   Ctor4 (profile):      (bot, bool pv, itr, ...)
//   Ctor5 (9-arg):        (nk, nullptr_t, cwe, rc)
//   Ctor7 (codegen guard): (nk, bool nf)        ← for codegen noFuse
//   Ctor8 (codegen g+bot): (nk, bot, bool nf)   ← for codegen guard+binOp
//   Ctor9 (guard+binOp):  (bot, bool nf)        ← for codegen guard without nk
#pragma once

#include <MNN/MNNDefine.h>
#include <vector>
#include <utility>
#include <cstddef>

namespace MNN {

struct ChainConstCheck {
    int chainPos = 0;              // 0-based position in the collected chain
    int inputIdx = 0;              // which input of that op is the const
    std::vector<float> values;     // expected const values (1e-4 tol)
    ChainConstCheck(int p, int in, std::vector<float> v)
        : chainPos(p), inputIdx(in), values(std::move(v)) {}
};

struct ExactPattern {
    std::vector<MNN::OpType> opTypes;
    int constElems;
    int constIndex;
    const char* ispType;
    const char* spvName;
    const char* namedKey;
    int convWeightElems = -1;
    MNN::BinaryOpOperation binOpType = MNN::BinaryOpOperation_ADD; // default: any BinaryOp
    std::vector<float> convWeightValues;   // if non-empty, require weight match (1e-4 tol)
    std::vector<float> constValues;        // if non-empty, require const blob match (1e-4 tol)
    // noFuse: match and consume the chain (advance scan) but KEEP ops primitive.
    // Used as longest-match guard so shorter generic patterns (e.g. auto_contrast)
    // never steal scalar control chains (algo_gamma / algo_cct).
    bool noFuse = false;
    // Per-position const-value checks: (chainPos, inputIdx) -> expected values.
    std::vector<ChainConstCheck> chainConstChecks;
    // Profile-variant disambiguation (blocks whose constants are graph Inputs):
    //   inputTrace[k]      — required producer type of ops[idx[0]]->inputIndexes[k]
    //                        (traced through ConvertTensor; -1 = any). Stops at Extra.
    //   inputMustBeInput   — input indices that must be fed directly by a graph Input op.
    //   nextOpType         — required type of the next non-CT/Const/Input op after the chain.
    //   chainBinOps        — (chainPos, BinaryOp sub-type) requirements at chain positions.
    std::vector<int> inputTrace;
    std::vector<int> inputMustBeInput;
    int nextOpType = -1;
    int nextBinOp = -1;   // required BinaryOp sub-type of the next op (-1 = any)
    std::vector<std::pair<int, MNN::BinaryOpOperation>> chainBinOps;
    bool requireConnectivity = true; // false for tree-shaped DAGs (calibration, histogram, tone_stats)

    // ── Ctor1: base (namedKey, constElems) ────────────────────────
    ExactPattern(std::vector<MNN::OpType> ops, int ce, int ci,
                 const char* isp, const char* spv, const char* nk = nullptr, int cwe = -1,
                 std::vector<float> cwv = {}, std::vector<float> cv = {})
        : opTypes(std::move(ops)), constElems(ce), constIndex(ci),
          ispType(isp), spvName(spv), namedKey(nk), convWeightElems(cwe),
          convWeightValues(std::move(cwv)), constValues(std::move(cv)) {}

    // ── Ctor2: binOp (binOpType, namedKey) ───────────────────────
    ExactPattern(std::vector<MNN::OpType> ops, int ce, int ci,
                 const char* isp, const char* spv, MNN::BinaryOpOperation bot,
                 const char* nk = nullptr, int cwe = -1,
                 std::vector<float> cwv = {}, std::vector<float> cv = {})
        : opTypes(std::move(ops)), constElems(ce), constIndex(ci),
          ispType(isp), spvName(spv), namedKey(nk), convWeightElems(cwe), binOpType(bot),
          convWeightValues(std::move(cwv)), constValues(std::move(cv)) {}

    // ── Ctor3: guard (bool nf, chainConstChecks) ─────────────────
    ExactPattern(std::vector<MNN::OpType> ops, int ce, int ci,
                 const char* isp, const char* spv, bool nf,
                 std::vector<ChainConstCheck> ccc = {})
        : opTypes(std::move(ops)), constElems(ce), constIndex(ci),
          ispType(isp), spvName(spv), noFuse(nf), chainConstChecks(std::move(ccc)) {}

    // ── Ctor4: profile-variant ────────────────────────────────────
    // The 7th arg is the bool marker so this ctor never collides with the
    // BinaryOp-sub-type ctor (whose 7th arg is the namedKey const char*).
    ExactPattern(std::vector<MNN::OpType> ops, int ce, int ci,
                 const char* isp, const char* spv, MNN::BinaryOpOperation bot,
                 bool pv, std::vector<int> itr, std::vector<int> imbi = {},
                 int nxt = -1, std::vector<std::pair<int, MNN::BinaryOpOperation>> cbo = {},
                 int nbo = -1)
        : opTypes(std::move(ops)), constElems(ce), constIndex(ci),
          ispType(isp), spvName(spv), binOpType(bot),
          inputTrace(std::move(itr)), inputMustBeInput(std::move(imbi)),
          nextOpType(nxt), nextBinOp(nbo), chainBinOps(std::move(cbo)) {
        (void)pv;
    }

    // ── Ctor5: 9-arg (nk, nullptr_t, cwe, requireConnectivity) ───
    ExactPattern(std::vector<MNN::OpType> ops, int ce, int ci,
                 const char* isp, const char* spv, const char* nk,
                 std::nullptr_t, int cwe, bool rc = true)
        : opTypes(std::move(ops)), constElems(ce), constIndex(ci),
          ispType(isp), spvName(spv), namedKey(nk), convWeightElems(cwe),
          requireConnectivity(rc) {}

    // ── Ctor7: codegen guard (namedKey + bool nf) ────────────────
    // Codegen emits (namedKey, true) for noFuse guard chains without binOp.
    // Matches entries like: ExactPattern({...}, -1, -1, "isp.noop_cct", "isp.noop_cct", nullptr, true)
    ExactPattern(std::vector<MNN::OpType> ops, int ce, int ci,
                 const char* isp, const char* spv, const char* nk,
                 bool nf)
        : opTypes(std::move(ops)), constElems(ce), constIndex(ci),
          ispType(isp), spvName(spv), namedKey(nk), noFuse(nf) {}

    // ── Ctor8: codegen guard + binOp (namedKey + bot + bool nf) ──
    // Codegen emits (namedKey, binOpType, true) for noFuse guard chains with binOp.
    // Matches entries like: ExactPattern({...}, -1, -1, "isp.noop_gamma", "isp.noop_gamma", nullptr, MNN::BinaryOpOperation_MUL, true)
    ExactPattern(std::vector<MNN::OpType> ops, int ce, int ci,
                 const char* isp, const char* spv, const char* nk,
                 MNN::BinaryOpOperation bot, bool nf)
        : opTypes(std::move(ops)), constElems(ce), constIndex(ci),
          ispType(isp), spvName(spv), namedKey(nk), binOpType(bot), noFuse(nf) {}

    // ── Ctor9: guard + binOp (no namedKey) ───────────────────────
    // Codegen emits (binOpType, true) for guard chains with binOp but no namedKey.
    ExactPattern(std::vector<MNN::OpType> ops, int ce, int ci,
                 const char* isp, const char* spv, MNN::BinaryOpOperation bot,
                 bool nf)
        : opTypes(std::move(ops)), constElems(ce), constIndex(ci),
          ispType(isp), spvName(spv), binOpType(bot), noFuse(nf) {}
};

} // namespace MNN
