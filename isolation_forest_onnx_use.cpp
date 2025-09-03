#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main() {
    // 1. ONNX Runtime 초기화
    const Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "isolation_forest");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // 2. ONNX 모델 로드
    const wchar_t* model_path = L"isolation_forest.onnx";
    Ort::Session session(env, model_path, session_options);

    // 3. 입력/출력 이름 가져오기
    std::vector<std::string> input_names_str = session.GetInputNames();
    std::vector<std::string> output_names_str = session.GetOutputNames();

    std::vector<const char*> input_names, output_names;
    for (auto& s : input_names_str) input_names.push_back(s.c_str());
    for (auto& s : output_names_str) output_names.push_back(s.c_str());

    std::cout << "INPUT: " << input_names[0] << std::endl;
    std::cout << "OUTPUT: ";
    for (auto& name : output_names) std::cout << name << " ";
    std::cout << std::endl;

    // 4. 입력 데이터 생성 (1x4 feature)
    std::vector<float> input_values = {0.5f, 1.2f, -0.3f, 1.1f};
    std::vector<int64_t> input_shape = {1, 4};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_values.data(), input_values.size(),
        input_shape.data(), input_shape.size());

    // 5. 모델 실행
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(), &input_tensor, input_names.size(),
        output_names.data(), output_names.size()
    );

    // 6. 결과 출력
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    std::cout << "INPUT feature: ";
    for (auto v : input_values) std::cout << v << " ";
    std::cout << std::endl;

    // 출력 텐서들 출력
    for (size_t i = 0; i < output_names.size(); i++) {
        std::cout << output_names[i] << " => ";
        float* output_data = output_tensors[i].GetTensorMutableData<float>();
        size_t output_num = output_tensors[i].GetTensorTypeAndShapeInfo().GetElementCount();
        for (size_t j = 0; j < output_num; j++) {
            std::cout << output_data[j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}



// int main() {
//     // Initialize ONNX Runtime
//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "isolation_forest");

//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);

//     // Load the ONNX model
//     const char* model_path = "isolation_forest.onnx";
//     Ort::Session session(env, model_path, session_options);

//     // Input data: 1 sample, 4 features
//     std::vector<float> input_values = {0.5f, 1.2f, -0.3f, 1.1f};
//     std::vector<int64_t> input_shape = {1, 4};

//     // Get input/output names
//     Ort::AllocatorWithDefaultOptions allocator;
//     char* input_name = session.GetInputName(0, allocator);
//     char* output_name = session.GetOutputName(0, allocator);

//     // Create input tensor
//     Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//         memory_info, input_values.data(), input_values.size(), input_shape.data(), input_shape.size());

//     // Run inference
//     std::vector<const char*> input_names = {input_name};
//     std::vector<const char*> output_names = {output_name};
//     auto output_tensors = session.Run(
//         Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);

//     // Get output
//     float* output_data = output_tensors[0].GetTensorMutableData<float>();
//     std::cout << "Isolation Forest output: " << output_data[0] << std::endl;

//     // Free resources
//     allocator.Free(input_name);
//     allocator.Free(output_name);
//     return 0;
// }