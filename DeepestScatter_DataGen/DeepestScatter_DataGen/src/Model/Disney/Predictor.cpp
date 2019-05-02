#include "Predictor.h"

#include "SceneSetup.pb.h"
#include "DisneyDescriptor.pb.h"
#include "ScatterSample.pb.h"
#include <optixu/optixu_math_namespace.h>
#include "Result.pb.h"

#include <torch/script.h>

namespace DeepestScatter
{
    DisneyPredictor::DisneyPredictor(const std::filesystem::path& model,
        std::shared_ptr<Dataset>& dataset)
    {

        std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(model.string());
        assert(module != nullptr);
        
        auto scene = dataset->getRecord<Persistance::SceneSetup>(0);
        auto sample = dataset->getRecord<Persistance::ScatterSample>(0);
        auto descriptor = dataset->getRecord<Persistance::DisneyDescriptor>(0);
        auto radiance = dataset->getRecord<Persistance::Result>(0).light_intensity();
        auto view = optix::make_float3(sample.view_direction().x(), sample.view_direction().y(), sample.view_direction().z());
        auto sun = optix::make_float3(scene.light_direction().x(), scene.light_direction().y(), scene.light_direction().z());

        float angle = acos(optix::dot(view, sun));
        std::cout << std::endl;
        std::vector<torch::jit::IValue> inputs;
        float zLayersBuffer[10][226];
        {
            size_t id = 0;
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 225; j++)
                {
                    zLayersBuffer[i][j] = static_cast<uint8_t>(descriptor.grid()[id]) / 256.0f;
                    id++;
                }
                zLayersBuffer[i][225] = angle;
            }
        }
        inputs.push_back(torch::from_blob(zLayersBuffer[0], { 1, 10, 226 }).cuda());

        at::Tensor output = module->forward(inputs).toTensor();

        std::cout << std::setprecision(16) << "== predicted radiance: " << output
            << " == real radiance: " << radiance << std::endl;
        exit(0);
    }
}
