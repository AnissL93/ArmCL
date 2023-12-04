/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/NEFunctions.h"

#include "src/common/cpuinfo/CpuInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/PoolManager.h"
#include "utils/CommonGraphOptions.h"
#include "utils/Utils.h"

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"

#include <chrono>

#define PRINT_VAL(x) std::cout << #x << " = " << x << std::endl;
using namespace arm_compute;
using namespace utils;
using namespace rapidjson;

class ParseJsonConfig
{
    private:
    bool parse_array(const Value &d, const char *key, int &a, int &b, int &c)
    {
        if(d.HasMember(key) && d[key].IsArray())
        {
            const rapidjson::Value &inputArray = d[key];

            // Iterate through the array elements
            a = inputArray[0].GetInt();
            b = inputArray[1].GetInt();
            c = inputArray[2].GetInt();
            return true;
        }
        else
        {
            std::cerr << "Not found key " << key << "\n";
            for(auto iter = d.MemberBegin(); iter != d.MemberEnd(); ++iter)
            {
                std::cout << iter->name.GetString() << std::endl;
            }
            return false;
        }
    }

    void parse_int_or_die(const Value &d, const char *key, int &result)
    {
        if(d.HasMember(key) && d[key].IsInt())
        {
            result = d[key].GetInt();
        }
        else
        {
            std::cerr << "Not found key " << key << "\n";
            exit(-1);
        }
    }

    public:
    bool parse_json(const std::string &json, std::string block_name = "", const std::string& type="expanded_conv")
    {
        std::ifstream inputFile(json.c_str());
        // Check if the file is open
        if(!inputFile.good())
        {
            std::cerr << "Error opening file: " << json << std::endl;
            return false; // Return an error code
        }

        // Read the file contents into a string
        std::string fileContent((std::istreambuf_iterator<char>(inputFile)),
                                std::istreambuf_iterator<char>());
        inputFile.close();

        rapidjson::Document d;
        d.Parse(fileContent.c_str());
        // Check for parsing errors
        if(d.HasParseError())
        {
            std::cerr << "Parse failed\n";
            return false;
        }

        if (block_name.empty()) {
            for (auto iter = d.MemberBegin(); iter != d.MemberEnd(); ++iter) {
                block_name = iter->name.GetString();
                const Value &p = d[block_name.c_str()];
                if (p["type"].GetString() == type) {
                    break ;
                }
            }

            std::cout << "Run the first item " << block_name << std::endl;
        }

        if (block_name.empty()) {
            std::cerr << "Can not parse type " << type;
            return false;
        }

        std::cout << "Run block " << block_name << std::endl;

        const Value &params = d[block_name.c_str()];

        // get block name

        if(!parse_array(params, "input", input_h, input_w, input_c))
        {
            std::cerr << "Parse input failed\n";
            return false;
        }

        if(!parse_array(params, "output", output_h, output_w, output_c))
        {
            std::cerr << "Parse output failed\n";
            return false;
        }
        parse_int_or_die(params, "expand", expand);
        parse_int_or_die(params, "kernel", kernel);
        parse_int_or_die(params, "stride", stride);
        parse_int_or_die(params, "idskip", idskip);
        parse_int_or_die(params, "pad_l", pad_l);
        parse_int_or_die(params, "pad_r", pad_r);

        std::cout << ">>>> Parameters " << std::endl;
        PRINT_VAL(type)
        PRINT_VAL(input_h)
        PRINT_VAL(input_w)
        PRINT_VAL(input_c)
        PRINT_VAL(output_h)
        PRINT_VAL(output_w)
        PRINT_VAL(output_c)
        PRINT_VAL(expand)
        PRINT_VAL(kernel)
        PRINT_VAL(stride)
        PRINT_VAL(idskip)
        PRINT_VAL(pad_l)
        PRINT_VAL(pad_r)

        return true;
    }

    int input_h, input_w, input_c, output_h, output_w, output_c, expand, kernel, stride, idskip, pad_l, pad_r, batch = 1;

    std::string block_name;
};

class NEONExpandedConvExample : public Example
{
    public:
    NEONExpandedConvExample()
        : cmd_parser()
    {
        block_config = cmd_parser.add_option<SimpleOption<std::string>>("config", "");
        block_config->set_help("Json config.");
        repeat_n = cmd_parser.add_option<SimpleOption<int>>("repeat_n", 50);
        repeat_n->set_help("Repeat times");
        block_name = cmd_parser.add_option<SimpleOption<std::string>>("name", "");
        block_name->set_help("The name of the block.");
        use_core = cmd_parser.add_option<SimpleOption<std::string>>("use_core", "both");
        use_core->set_help("Set using core: big, small or both");
    }

    CommandLineParser          cmd_parser;
    SimpleOption<std::string> *block_config{ nullptr };
    SimpleOption<int>* repeat_n{nullptr};
    SimpleOption<std::string> * block_name{nullptr};
    SimpleOption<std::string> * use_core{nullptr};

    void init_tensor_c(Tensor &t, int c, DataType dt = DataType::F32)
    {
        TensorShape shape(c);
        t.allocator()->init(TensorInfo(shape, 1, dt));
    }
    void init_tensor_nchw(Tensor &t, int w, int h, int c, int n, DataType dt = DataType::F32)
    {
        DataLayout  layout = DataLayout::NCHW;
        TensorShape shape;
        shape.set(graph::get_dimension_idx(layout, DataLayoutDimension::BATCHES), n);
        shape.set(graph::get_dimension_idx(layout, DataLayoutDimension::CHANNEL), c);
        shape.set(graph::get_dimension_idx(layout, DataLayoutDimension::WIDTH), w);
        shape.set(graph::get_dimension_idx(layout, DataLayoutDimension::HEIGHT), h);
        t.allocator()->init(TensorInfo(shape, 1, dt, layout));
    }

    void init_tensor_chw(Tensor &t, int h, int w, int c, DataType dt = DataType::F32)
    {
        DataLayout  layout = DataLayout::NCHW;
        TensorShape shape;
        shape.set(graph::get_dimension_idx(layout, DataLayoutDimension::CHANNEL), c);
        shape.set(graph::get_dimension_idx(layout, DataLayoutDimension::WIDTH), w);
        shape.set(graph::get_dimension_idx(layout, DataLayoutDimension::HEIGHT), h);
        t.allocator()->init(TensorInfo(shape, 1, dt));
    }

    void get_expanded_conv_float_tensors()
    {
        int exp_size = config.input_c * config.expand;
        // allocate all weights
        init_tensor_nchw(weights_conv0, 1, 1, config.input_c, exp_size);
        init_tensor_chw(weights_dwconv, config.kernel, config.kernel, exp_size);
        init_tensor_nchw(weights_conv1, 1, 1, exp_size, config.output_c);
        // allocate for bn
        init_tensor_c(mean0, exp_size);
        init_tensor_c(var0, exp_size);
        init_tensor_c(gamma0, exp_size);
        init_tensor_c(beta0, exp_size);
        init_tensor_c(mean1, exp_size);
        init_tensor_c(var1, exp_size);
        init_tensor_c(gamma1, exp_size);
        init_tensor_c(beta1, exp_size);
        init_tensor_c(mean2, config.output_c);
        init_tensor_c(var2, config.output_c);
        init_tensor_c(gamma2, config.output_c);
        init_tensor_c(beta2, config.output_c);

        // allocate inputs and outputs
        init_tensor_nchw(out_conv0, config.input_w, config.input_h, exp_size, config.batch);
        init_tensor_nchw(out_act0, config.input_w, config.input_h, exp_size, config.batch);
        init_tensor_nchw(out_bn0, config.input_w, config.input_h, exp_size, config.batch);

        init_tensor_nchw(out_dwconv, config.output_w, config.output_h, exp_size, config.batch);
        init_tensor_nchw(out_act1, config.output_w, config.output_h, exp_size, config.batch);
        init_tensor_nchw(out_bn1, config.output_w, config.output_h, exp_size, config.batch);

        init_tensor_nchw(out_conv1, config.output_w, config.output_h, config.output_c, config.batch);
        init_tensor_nchw(out_act2, config.output_w, config.output_h, config.output_c, config.batch);
        init_tensor_nchw(out_bn2, config.output_w, config.output_h, config.output_c, config.batch);

        if(config.idskip != 0)
        {
            init_tensor_nchw(out_add, config.output_w, config.output_h, config.output_c, config.batch);
        }
    }

    void configure_layers()
    {
        conv0->configure(&src, &weights_conv0, nullptr, &out_conv0, PadStrideInfo(1, 1));
        bn0.configure(&out_conv0, &out_bn0, &mean0, &var0, &beta0, &gamma0);
        act0.configure(&out_bn0, &out_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        dwconv->configure(&out_act0, &weights_dwconv, nullptr, &out_dwconv,
                          PadStrideInfo(config.stride, config.stride, config.pad_l, config.pad_r, config.pad_l, config.pad_r, DimensionRoundingType::CEIL));
        bn1.configure(&out_dwconv, &out_bn1, &mean1, &var1, &beta1, &gamma1);
        act1.configure(&out_bn1, &out_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        conv1->configure(&out_act1, &weights_conv1, nullptr, &out_conv1, PadStrideInfo(1, 1));
        bn2.configure(&out_conv1, &out_bn2, &mean2, &var2, &beta2, &gamma2);
        act2.configure(&out_bn2, &out_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        if(config.idskip != 0)
        {
            add.configure(&src, &out_bn2, &out_add, ConvertPolicy::WRAP);
        }
    }

    bool do_setup(int argc, char **argv) override {
        cmd_parser.parse(argc, argv);
        cmd_parser.validate();

        // set core
        if (use_core->value() == "small") {
            cpuinfo::g_use_core = cpuinfo::UseCore::SMALL;
        } else if (use_core->value() == "big") {
            cpuinfo::g_use_core = cpuinfo::UseCore::BIG;
        }

        std::string config_file = block_config->value();
        std::cout << "config file: " << config_file << std::endl;

        if(!config.parse_json(config_file, block_name->value()))
        {
            std::cerr << "Parse config file failed\n";
            return false;
        }

        // Create memory manager components
        // We need 2 memory managers: 1 for handling the tensors within the functions (mm_layers) and 1 for handling the input and output tensors of the functions (mm_transitions))
        auto lifetime_mgr0  = std::make_shared<BlobLifetimeManager>();                           // Create lifetime manager
        auto lifetime_mgr1  = std::make_shared<BlobLifetimeManager>();                           // Create lifetime manager
        auto pool_mgr0      = std::make_shared<PoolManager>();                                   // Create pool manager
        auto pool_mgr1      = std::make_shared<PoolManager>();                                   // Create pool manager
        auto mm_layers      = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr0, pool_mgr0); // Create the memory manager
        auto mm_transitions = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr1, pool_mgr1); // Create the memory manager

        // The weights and biases tensors should be initialized with the values inferred with the training

        // Set memory manager where allowed to manage internal memory requirements
        conv0  = std::make_unique<NEConvolutionLayer>(mm_layers);
        conv1  = std::make_unique<NEConvolutionLayer>(mm_layers);
        dwconv = std::make_unique<NEDepthwiseConvolutionLayer>(mm_layers);

        const TensorShape src_shape(config.input_w, config.input_h, config.input_c, config.batch);
        src.allocator()->init(TensorInfo(src_shape, 1, DataType::F32));

        get_expanded_conv_float_tensors();
        configure_layers();

        // We need 2 memory groups for handling the input and output
        // We call explicitly allocate after manage() in order to avoid overlapping lifetimes
        memory_group0 = std::make_unique<MemoryGroup>(mm_transitions);
        memory_group1 = std::make_unique<MemoryGroup>(mm_transitions);

        auto memory = [&](Tensor &t, int mem_id)
        {
            if(mem_id == 0)
            {
                memory_group0->manage(&t);
                t.allocator()->allocate();
            }
            else
            {
                memory_group1->manage(&t);
                t.allocator()->allocate();
            }
        };

        memory(out_conv0, 0);
        memory(out_bn0, 1);
        memory(out_act0, 0);
        memory(out_dwconv, 1);
        memory(out_bn1, 0);
        memory(out_act1, 1);
        memory(out_conv1, 0);
        memory(out_bn2, 1);
        memory(out_act2, 0);
        if(config.idskip != 0)
        {
            memory(out_add, 1);
        }

        // Now that the padding requirements are known we can allocate all tensors
        src.allocator()->allocate();
        weights_conv0.allocator()->allocate();
        weights_dwconv.allocator()->allocate();
        weights_conv1.allocator()->allocate();
        mean0.allocator()->allocate();
        mean1.allocator()->allocate();
        mean2.allocator()->allocate();
        var0.allocator()->allocate();
        var1.allocator()->allocate();
        var2.allocator()->allocate();
        gamma0.allocator()->allocate();
        gamma1.allocator()->allocate();
        gamma2.allocator()->allocate();
        beta0.allocator()->allocate();
        beta1.allocator()->allocate();
        beta2.allocator()->allocate();

        /* -----------------------End: [Allocate tensors] */

        // Populate the layers manager. (Validity checks, memory allocations etc)
        mm_layers->populate(allocator, 1 /* num_pools */);

        // Populate the transitions manager. (Validity checks, memory allocations etc)
        mm_transitions->populate(allocator, 2 /* num_pools */);

        return true;
    }

    void do_run() override
    {
        // Acquire memory for the memory groups
        memory_group0->acquire();
        memory_group1->acquire();

        auto start_time = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < repeat_n->value(); ++i)
        {
            conv0->run();
            bn0.run();
            act0.run();
            dwconv->run();
            bn1.run();
            act1.run();
            conv1->run();
            bn2.run();
            act2.run();
            if(config.idskip != 0)
            {
                add.run();
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        std::cout << "Total time (" << repeat_n->value() << " times): " << duration.count() << " ms" << std::endl;
        std::cout << "Average time: " << duration.count() / (double)repeat_n->value() << " ms" << std::endl;
        // Release memory
        memory_group0->release();
        memory_group1->release();
    }

    private:
    // The src tensor should contain the input image

    ParseJsonConfig config;

    Tensor src{};

    // Intermediate tensors used
    Tensor weights_conv0{};
    Tensor weights_dwconv{};
    Tensor weights_conv1{};
    Tensor mean0{}, var0{}, gamma0{}, beta0{};
    Tensor mean1{}, var1{}, gamma1{}, beta1{};
    Tensor mean2{}, var2{}, gamma2{}, beta2{};

    Tensor out_conv0{};
    Tensor out_act0{};
    Tensor out_bn0{};

    Tensor out_dwconv{};
    Tensor out_act1{};
    Tensor out_bn1{};

    Tensor out_conv1{};
    Tensor out_act2{};
    Tensor out_bn2{};

    Tensor out_add{};

    // Allocator
    Allocator allocator{};

    // Memory groups
    std::unique_ptr<MemoryGroup> memory_group0{};
    std::unique_ptr<MemoryGroup> memory_group1{};

    // Layers
    std::unique_ptr<NEConvolutionLayer>          conv0{};
    std::unique_ptr<NEDepthwiseConvolutionLayer> dwconv{};
    std::unique_ptr<NEConvolutionLayer>          conv1{};
    NEActivationLayer                            act0{};
    NEActivationLayer                            act1{};
    NEActivationLayer                            act2{};
    NEBatchNormalizationLayer                    bn0{};
    NEBatchNormalizationLayer                    bn1{};
    NEBatchNormalizationLayer                    bn2{};
    NEArithmeticAddition                         add{};
};

/** Main program for cnn test
 *
 * The example implements the following CNN architecture:
 *
 * Input -> conv0:5x5 -> act0:relu -> pool:2x2 -> conv1:3x3 -> act1:relu -> pool:2x2 -> fc0 -> act2:relu -> softmax
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return utils::run_example<NEONExpandedConvExample>(argc, argv);
}
