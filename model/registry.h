#pragma once

#include "transformer.h"

std::shared_ptr<Transformer> llama38b(int VocabSize = 32000,
                                      float NormEps = 1e-5,
                                      float RopeTheta = 10000.0,
                                      bool RopeTraditional = false);

std::shared_ptr<Transformer> llama27bChat(int VocabSize = 32000,
                                          float NormEps = 1e-5,
                                          float RopeTheta = 10000.0,
                                          bool RopeTraditional = false);

std::shared_ptr<Transformer> tinyLlama11BChatV10(int VocabSize = 32000,
                                                 float NormEps = 1e-5,
                                                 float RopeTheta = 10000.0,
                                                 bool RopeTraditional = false);