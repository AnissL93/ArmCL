/*
 * Copyright (c) 2019-2021, 2023 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEBatchToSpaceLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/BatchToSpaceDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/BatchToSpaceLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(BatchToSpaceLayer)

template <typename T>
using NEBatchToSpaceLayerFixture = BatchToSpaceLayerValidationFixture<Tensor, Accessor, NEBatchToSpaceLayer, T>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(ValidateStatic, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(16U, 8U, 2U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(16U, 8U, 2U, 16U), 1, DataType::F32),    // Supported: blockx != blocky && blockx > blocky
                                                       TensorInfo(TensorShape(16U, 8U, 2U, 16U), 1, DataType::F32),    // Supported: blockx != blocky && blocky > blockx
                                                       TensorInfo(TensorShape(16U, 8U, 2U, 4U), 1, DataType::F32),     // Invalid: Mismatching data types
                                                       TensorInfo(TensorShape(16U, 8U, 2U, 4U), 1, DataType::F32),     // Invalid: Negative block shapes
                                                       TensorInfo(TensorShape(32U, 16U, 2U, 4U, 4U), 1, DataType::F32),// Unsupported tensor rank
                                                       TensorInfo(TensorShape(16U, 8U, 2U, 16U), 1, DataType::F32),    // Invalid output tensor shape (invalid batch dimension)
                                                       TensorInfo(TensorShape(16U, 8U, 2U, 16U), 1, DataType::F32),    // Invalid output tensor shape (invalid spatial dimension)
                                                       TensorInfo(TensorShape(16U, 8U, 2U, 16U), 1, DataType::F32),    // Supported: correct tensor shape with cropping
                                                       TensorInfo(TensorShape(16U, 8U, 2U, 16U), 1, DataType::F32),    // Invalid tensor shape with cropping
                                                     }),
               framework::dataset::make("BlockShapeX", { 2, 4, 2, 2, 2, 2, 2, 2, 2, 2 })),
               framework::dataset::make("BlockShapeY", { 2, 2, 4, 2, -2, 2, 2, 2, 2, 2 })),
               framework::dataset::make("CropInfo", {
                CropInfo{}, CropInfo{}, CropInfo{}, CropInfo{}, CropInfo{}, CropInfo{}, CropInfo{}, CropInfo{}, CropInfo{3, 2, 1, 3}, CropInfo{3, 2, 1, 3}
               })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 16U, 2U, 1U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(64U, 16U, 2U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 32U, 2U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 16U, 2U, 1U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(32U, 16U, 2U, 1U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 8U, 2U, 1U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 16U, 2U, 1U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(33U, 32U, 2U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27, 12U, 2U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 16U, 2U, 4U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("Expected", { true, true, true, false, false, false, false, false, true, false})),
               input_info, block_shape_x, block_shape_y, crop_info, output_info, expected)
{
    bool has_error = bool(NEBatchToSpaceLayer::validate(&input_info.clone()->set_is_resizable(false), block_shape_x, block_shape_y, &output_info.clone()->set_is_resizable(false), crop_info));
    ARM_COMPUTE_EXPECT(has_error == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEBatchToSpaceLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallBatchToSpaceLayerDataset(), framework::dataset::make("DataType",
                                                                                                                       DataType::F32)),
                                                                                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunSmallWithCropping, NEBatchToSpaceLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(datasets::SmallBatchToSpaceLayerWithCroppingDataset(), framework::dataset::make("DataType",
                                                                                                                       DataType::F32)),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEBatchToSpaceLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeBatchToSpaceLayerDataset(), framework::dataset::make("DataType",
                                                                                                                     DataType::F32)),
                                                                                                             framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEBatchToSpaceLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallBatchToSpaceLayerDataset(), framework::dataset::make("DataType",
                                                                                                                      DataType::F16)),
                                                                                                              framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunSmallWithCropping, NEBatchToSpaceLayerFixture<half>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(datasets::SmallBatchToSpaceLayerWithCroppingDataset(), framework::dataset::make("DataType",
                                                                                                                       DataType::F16)),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEBatchToSpaceLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeBatchToSpaceLayerDataset(), framework::dataset::make("DataType",
                                                                                                                    DataType::F16)),
                                                                                                            framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END() // BatchToSpace
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
