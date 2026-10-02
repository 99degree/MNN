//
//  PostConverter.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <MNN/MNNDefine.h>
#include "MNN_generated.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/util.h"
#include "config.hpp"

/**
 *@brief optimize MNN net
 */
std::unique_ptr<MNN::NetT> optimizeNet(std::unique_ptr<MNN::NetT>& netT, bool forTraining, modelConfig& config, const std::vector<std::string>& expectPasses);

/**
 *@brief Ensure template merge registrations are initialized
 * Call this before using TemplateMerge::getInstance() to avoid
 * static initialization order issues across shared libraries.
 */
namespace MNN {
namespace Express {
MNN_PUBLIC void __attribute__((visibility("default"))) ensureBinaryAddToEltwiseRegistered();
} // namespace Express
} // namespace MNN

#endif // OPTIMIZER_HPP
