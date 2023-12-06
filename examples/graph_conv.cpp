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

#include "arm_compute/graph.h"
#include "support/ToolchainSupport.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

using namespace arm_compute;
using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;


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
    bool parse_array(const Value &d, const char *key, std::vector<int>& vals)
    {
        if(d.HasMember(key) && d[key].IsArray())
        {
            const rapidjson::Value &inputArray = d[key];

            size_t size = inputArray.Size();
            assert (inputArray.IsArray());
            for(size_t i = 0; i < size; ++i) {
                vals.push_back(inputArray[i].GetInt());
            }
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
    bool parse_json(const std::string &json, std::string block_name)
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
            std::cerr << "Can not parse type " << type;
            return false;
        }

        std::cout << "Run block " << block_name << std::endl;

        const Value &params = d[block_name.c_str()];

        type = params["type"].GetString();
        PRINT_VAL(type);

        // get block name
        std::vector<int> vals;
        if(!parse_array(params, "input", vals))
        {
            std::cerr << "Parse input failed\n";
            return false;
        }
        input_h = vals[0];
        input_w = vals[1];
        input_c = vals[2];

        vals.clear();
        if(!parse_array(params, "output", vals))
        {
            std::cerr << "Parse output failed\n";
            return false;
        }
        printf("Value number %d", vals.size());
        if (type != "Logits") {
            output_h = vals[0];
            output_w = vals[1];
            output_c = vals[2];
        } else {
            output_h = 1;
            output_w = 1;
            output_c = vals[0];
        }

        std::cout << ">>>> Parameters " << std::endl;
        PRINT_VAL(type)
        PRINT_VAL(input_h)
        PRINT_VAL(input_w)
        PRINT_VAL(input_c)
        PRINT_VAL(output_h)
        PRINT_VAL(output_w)
        PRINT_VAL(output_c)
        return true;
    }

    int input_h, input_w, input_c, output_h, output_w, output_c, batch = 1;

    std::string block_name;
    std::string type;
};

class GraphConvExample : public Example
{
    public:
    GraphConvExample()
        : cmd_parser(), common_opts(cmd_parser), graph(0, "FirstConv")
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
    CommonGraphOptions         common_opts;
    CommonGraphParams          common_params;

    SimpleOption<std::string> *block_config{ nullptr };
    SimpleOption<int>* repeat_n{nullptr};
    SimpleOption<std::string> * block_name{nullptr};
    SimpleOption<std::string> * use_core{nullptr};

    bool do_setup(int argc, char **argv) override {
        cmd_parser.parse(argc, argv);
        cmd_parser.validate();

        common_params = consume_common_graph_parameters(common_opts);

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

        // Return when help menu is requested
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Print parameter values
        std::cout << common_params << std::endl;


        // Create input descriptor
        const TensorShape tensor_shape     = TensorShape(config.input_w, config.input_h, config.input_c, common_params.batches);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        graph << common_params.target
              << common_params.fast_math_hint;

        if (config.type == "Conv") {
            create_first_conv_graph(input_descriptor);
        } else if (config.type == "Conv_1") {
            create_conv1_graph(input_descriptor);
        } else if (config.type == "Logits") {
            create_logits(input_descriptor);
        } else {
            printf("Unsupported block type \n");
            exit(-1);
        }

        int output_size = common_params.batches * config.output_w * config.output_h * config.output_c;
        graph << OutputLayer(std::make_unique<DummyAccessor>(output_size));

        // Finalize graph
        GraphConfig config;
        config.num_threads = common_params.threads;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_mode  = common_params.tuner_mode;
        config.tuner_file  = common_params.tuner_file;
        config.mlgo_file   = common_params.mlgo_file;

        graph.finalize(common_params.target, config);
        return true;
    }

    void create_first_conv_graph(TensorDescriptor& input_descriptor) {
        graph << InputLayer(input_descriptor, get_random_accessor(0.f, 1.f))
              << ConvolutionLayer(3U, 3U, 32U,
                                  get_random_accessor(0.f, 1.f),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                  PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL))
                  .set_name("Conv")
              << BatchNormalizationLayer(
                     get_random_accessor(0.f, 1.f),
                     get_random_accessor(0.f, 1.f),
                     get_random_accessor(0.f, 1.f),
                     get_random_accessor(0.f, 1.f),
                                         0.0010000000474974513f)
                  .set_name("Conv/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.))
                  .set_name("Conv/Relu6");
    }

    void create_conv1_graph(TensorDescriptor& input_descriptor) {
        graph << InputLayer(input_descriptor, get_random_accessor(0.f, 1.f))
              << ConvolutionLayer(1U, 1U, config.output_c,
                                  get_random_accessor(0.f, 1.f),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                  PadStrideInfo(1, 1, 0, 0, 0, 0, DimensionRoundingType::CEIL))
                  .set_name("Conv")
              << BatchNormalizationLayer(
                  get_random_accessor(0.f, 1.f),
                  get_random_accessor(0.f, 1.f),
                  get_random_accessor(0.f, 1.f),
                  get_random_accessor(0.f, 1.f),
                  0.0010000000474974513f)
                  .set_name("Conv/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.))
                  .set_name("Conv/Relu6");
    }

    void create_logits(TensorDescriptor& input_descriptor) {
        graph << InputLayer(input_descriptor, get_random_accessor(0.f, 1.f))
            << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, DataLayout::NCHW)).set_name("Logits/AvgPool")
            << ConvolutionLayer(1U, 1U, config.output_c,
                                  get_random_accessor(0., 1.),
                                  get_random_accessor(0., 1.),
                                PadStrideInfo(1, 1, 0, 0))
                .set_name("Logits/Conv2d_1c_1x1");
    }


    void do_run() override
    {
        // Acquire memory for the memory groups
        auto &gm = graph.gm();
        auto& workloads = gm.get_workloads(graph.graph());

        gm.prepare_inputs(workloads);

        auto start_time = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < repeat_n->value(); ++i)
        {
            gm.execute_tasks(workloads);
        }

        auto end_time = std::chrono::high_resolution_clock::now();

        gm.prepare_outputs(workloads);
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        std::cout << "Total time (" << repeat_n->value() << " times): " << duration.count() << " ms" << std::endl;
        std::cout << "Average time: " << duration.count() / (double)repeat_n->value() << " ms" << std::endl;
    }

    private:
    // The src tensor should contain the input image
    ParseJsonConfig config;
    Stream             graph;

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
    return utils::run_example<GraphConvExample>(argc, argv);
}
