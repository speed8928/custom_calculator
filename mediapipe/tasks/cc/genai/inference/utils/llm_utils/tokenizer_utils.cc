// Copyright 2024 The MediaPipe Authors.
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

#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/tokenizer_utils.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "src/sentencepiece_model.pb.h"

namespace mediapipe::tasks::genai::llm_utils {

// Returns a valid vocab ModelProto with the targeted vocab size.
std::string GetFakeSerializedVocabProto(int vocab_size) {
  sentencepiece::ModelProto model_proto;
  for (int i = 0; i < vocab_size; ++i) {
    auto *sp1 = model_proto.add_pieces();
    if (i == 0) {
      // A valid vocab proto needs to have one and only one UNKNOWN token.
      sp1->set_type(sentencepiece::ModelProto::SentencePiece::UNKNOWN);
      sp1->set_piece(absl::StrCat(i));
    } else {
      sp1->set_type(sentencepiece::ModelProto::SentencePiece::NORMAL);
      sp1->set_piece(absl::StrCat(i));
    }
  }
  return model_proto.SerializeAsString();
}

}  // namespace mediapipe::tasks::genai::llm_utils
