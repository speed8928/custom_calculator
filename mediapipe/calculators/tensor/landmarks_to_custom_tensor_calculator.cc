// Copyright 2021 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/calculators/tensor/landmarks_to_custom_tensor_calculator.h"

#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "mediapipe/calculators/tensor/landmarks_to_custom_tensor_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
    namespace api2 {

// Returns the scale attribute should be multiplied by.
            float GetAttributeScale(
                    const LandmarksToCustomTensorCalculatorOptions::Attribute& attribute,
                    const std::pair<int, int>& image_size) {
                switch (attribute) {
                    case LandmarksToCustomTensorCalculatorOptions::X:
                    case LandmarksToCustomTensorCalculatorOptions::Z:
                        return image_size.first;
                    case LandmarksToCustomTensorCalculatorOptions::Y:
                        return image_size.second;
                    case LandmarksToCustomTensorCalculatorOptions::VISIBILITY:
                    case LandmarksToCustomTensorCalculatorOptions::PRESENCE:
                        return 1.0f;
                }
            }

            template <typename LandmarkType>
            float GetAttribute(
                    const LandmarkType& landmark,
                    const LandmarksToCustomTensorCalculatorOptions::Attribute& attribute) {
                switch (attribute) {
                    case LandmarksToCustomTensorCalculatorOptions::X:
                        return landmark.x();
                    case LandmarksToCustomTensorCalculatorOptions::Y:
                        return landmark.y();
                    case LandmarksToCustomTensorCalculatorOptions::Z:
                        return landmark.z();
                    case LandmarksToCustomTensorCalculatorOptions::VISIBILITY:
                        return landmark.visibility();
                    case LandmarksToCustomTensorCalculatorOptions::PRESENCE:
                        return landmark.presence();
                }
            }






        class LandmarksToCustomTensorCalculatorImpl
                : public NodeImpl<LandmarksToCustomTensorCalculator> {
            public:

            absl::Status Open(CalculatorContext* cc) override {
                //if (cc->Service(kMemoryManagerService).IsAvailable()) {
                //    memory_manager_ = &cc->Service(kMemoryManagerService).GetObject();
                //}
                options_ = cc->Options<LandmarksToCustomTensorCalculatorOptions>();
                RET_CHECK(options_.attributes_size() > 0)
                        << "At least one attribute must be specified";

                RET_CHECK(kInLandmarkList(cc).IsConnected() ^
                          kInNormLandmarkList(cc).IsConnected())
                        << "Exactly one landmarks input should be provided";
                RET_CHECK_EQ(kInNormLandmarkList(cc).IsConnected(),
                             kImageSize(cc).IsConnected())
                        << "Image size should be provided only for normalized landmarks";



                // Fill tensor with landmark attributes.

                return absl::OkStatus();
            }



            absl::Status Process(CalculatorContext* cc) override {

                const NormalizedLandmarkList& landmarks = kInNormLandmarkList(cc).Get();
                int bodyparts [] = {0,12,14,16,11,13,15,24,26,28,23,25,27};
                for (int i: bodyparts){
                    const mediapipe::NormalizedLandmark landmark = landmarks.landmark(i);


                    if (idx == 0)
                    {
                        v.push_back(landmark.x());
                        v.push_back(landmark.y());
                        v.push_back(0);
                        v.push_back(0);
                        x_previous = landmark.x();
                        y_previous = landmark.y();

                    }
                    else
                    {
                        float x_vel = (landmark.x() - x_previous)/0.01;
                        float y_vel = (landmark.y() - y_previous)/0.01;
                        v.push_back(landmark.x());
                        v.push_back(landmark.y());
                        v.push_back(x_vel);
                        v.push_back(y_vel);
                        x_previous = landmark.x();
                        y_previous = landmark.y();
                    }
                }
                item.push_back(v);

                Tensor tensor(Tensor::ElementType::kFloat32, {30,52});
                auto* buffer = tensor.GetCpuWriteView().buffer<float>();
                if (idx>=30)
                {
                    std::move(item.begin()+1,item.end(),item.begin());
                    item.pop_back();
                    //std::cout<<item[i][j]<<std::endl;

                    for (int i = 0; i < 30; ++i) {
                        for (int j = 0; j < 52; ++j) {
                            buffer[i * 30 + j] = item[i][j];
                        }
                    }

                }


                v.erase (v.begin(),v.end());
                idx++;

                // Convert landmarks to tensor.
                auto result = std::vector<Tensor>();
                result.push_back(std::move(tensor));
                kOutTensors(cc).Send(std::move(result));

                return absl::OkStatus();
            }

//            static absl::Status UpdateContract(CalculatorContract* cc) {
//                cc->UseService(kMemoryManagerService).Optional();
//                return absl::OkStatus();
//            }



            private:
            LandmarksToCustomTensorCalculatorOptions options_;
            // Enable pooling of AHWBs in Tensor instances.

            int idx = 0;
            float x_previous = 0;
            float y_previous = 0;

            std::vector<float> v;
            std::vector< std::vector< float > > item;

        };


        MEDIAPIPE_NODE_IMPLEMENTATION(LandmarksToCustomTensorCalculatorImpl);

    }  // namespace api2
}  // namespace mediapipe