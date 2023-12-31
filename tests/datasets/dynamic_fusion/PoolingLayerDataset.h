/*
 * Copyright (c) 2023 Arm Limited.
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
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "utils/TypePrinter.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/Pool2dAttributes.h"


using Pool2dAttributes = arm_compute::experimental::dynamic_fusion::Pool2dAttributes;

namespace arm_compute
{
namespace test
{
namespace datasets
{

class DynamicFusionPoolingLayerDataset
{
public:
    using type = std::tuple<TensorShape, Pool2dAttributes>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator      src_it,
                 std::vector<Pool2dAttributes>::const_iterator infos_it)
            : _src_it{ std::move(src_it) },
              _infos_it{ std::move(infos_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_src_it << ":";
            description << "Info=" << *_infos_it << ":";
            return description.str();
        }

        DynamicFusionPoolingLayerDataset::type operator*() const
        {
            return std::make_tuple(*_src_it, *_infos_it);
        }

        iterator &operator++()
        {
            ++_src_it;
            ++_infos_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator      _src_it;
        std::vector<Pool2dAttributes>::const_iterator _infos_it;
    };

    iterator begin() const
    {
        return iterator(_src_shapes.begin(), _infos.begin());
    }

    int size() const
    {
        return std::min(_src_shapes.size(), _infos.size());
    }

    void add_config(TensorShape src, Pool2dAttributes info)
    {
        _src_shapes.emplace_back(std::move(src));
        _infos.emplace_back(std::move(info));
    }

protected:
    DynamicFusionPoolingLayerDataset()                       = default;
    DynamicFusionPoolingLayerDataset(DynamicFusionPoolingLayerDataset &&) = default;

private:
    std::vector<TensorShape>      _src_shapes{};
    std::vector<Pool2dAttributes> _infos{};
};

// Special pooling dataset
class PoolingLayerDatasetSpecialDynamicFusion final : public DynamicFusionPoolingLayerDataset
{
public:
    PoolingLayerDatasetSpecialDynamicFusion()
    {
        // NCHW DataLayout 
        // Special cases
        add_config(TensorShape(2U, 3U, 4U, 1U), Pool2dAttributes().pool_type(PoolingType::AVG).pool_size(Size2D(2,2)).stride(Size2D(3,3)));
        add_config(TensorShape(60U, 52U, 3U, 2U), Pool2dAttributes().pool_type(PoolingType::AVG).pool_size(Size2D(100,100)).stride(Size2D(5,5)).pad(Padding2D(50,50,50,50)));
        // Asymmetric padding
        add_config(TensorShape(112U, 112U, 32U), Pool2dAttributes().pool_type(PoolingType::MAX).pool_size(Size2D(3,3)).pad(Padding2D(0,1,0,1)).stride(Size2D(2,2)));
        add_config(TensorShape(14U, 14U, 832U), Pool2dAttributes().pool_type(PoolingType::MAX).pool_size(Size2D(2,2)).stride(Size2D(1,1)).pad(Padding2D(0,0,0,0)));

    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute