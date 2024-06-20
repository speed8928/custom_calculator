#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

namespace {

inline float Sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

}  // namespace

// Input:
//  TENSORS - Vector of Tensors of type kFloat32. Only the first tensor will be
//  used. This will output of the action recognition model. The tensor should
//  1 x 4.
//

class VerifyModelOutputCalculator : public Node {
 public:
  static constexpr Input<std::vector<Tensor>> kInTensors{"TENSORS"};
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
   
  MEDIAPIPE_NODE_CONTRACT(kInTensors);

 private:
  int num_landmarks_ = 0;
};
MEDIAPIPE_REGISTER_NODE(VerifyModelOutputCalculator);

absl::Status VerifyModelOutputCalculator::Open(CalculatorContext* cc) {
  return absl::OkStatus();
}

absl::Status VerifyModelOutputCalculator::Process(CalculatorContext* cc) {
  if (kInTensors(cc).IsEmpty()) {
    return absl::OkStatus();
  }

  const auto& input_tensors = *kInTensors(cc);
  RET_CHECK(input_tensors[0].element_type() == Tensor::ElementType::kFloat32);
  int num_values = input_tensors[0].shape().num_elements();
  
  std::vector<int> model_output_dims = input_tensors[0].shape().dims;

  LOG(INFO) << "model output: " << model_output_dims;

  auto view = input_tensors[0].GetCpuReadView();
  auto raw_landmarks = view.buffer<float>();
  
  // Assuming the tensor is 1 x 4
  int rows = model_output_dims[0];
  int cols = model_output_dims[1];

  // Traverse the tensor
  for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
          int index = i * cols + j;
          float value = raw_landmarks[index];
          std::cout << "Element at (" << i << ", " << j << "): " << value << std::endl;
      }
  }

  return absl::OkStatus();
}

}  // namespace api2
}  // namespace mediapipe
