#include "decoding.h"
#include "base.h"
#include "tokenizer.h"
#include <algorithm>
#include <cassert>
#include <mlx/ops.h>
#include <sstream>
#include <zlib.h>

namespace whisper {

// Utility function implementation
float compressionRatio(const std::string &Text) {
  if (Text.empty()) return 1.0f;
  
  std::vector<uint8_t> Input(Text.begin(), Text.end());
  uLongf CompressedSize = compressBound(Input.size());
  std::vector<uint8_t> Compressed(CompressedSize);
  
  int Result = compress(Compressed.data(), &CompressedSize, 
                       Input.data(), Input.size());
  
  if (Result != Z_OK) return 1.0f;
  
  return static_cast<float>(Input.size()) / static_cast<float>(CompressedSize);
}

// Language detection implementation
std::pair<mx::array, std::vector<std::map<std::string, float>>>
detectLanguage(std::shared_ptr<Whisper> Model, const mx::array &Mel,
               std::shared_ptr<Tokenizer> Tokenizer) {
  
  if (!Tokenizer) {
    Tokenizer = getTokenizer(Model->isMultilingual(), Model->numLanguages());
  }
  
  if (!Tokenizer->Language || 
      std::find(Tokenizer->SotSequence.begin(), 
                Tokenizer->SotSequence.end(), 
                Tokenizer->languageToken()) == Tokenizer->SotSequence.end()) {
    throw std::runtime_error("This model doesn't have language tokens so it can't perform lang id");
  }
  debugArray(Mel, "detectLanguage Mel");
  bool Single = Mel.ndim() == 2;
  mx::array MelArray = Single ? mx::expand_dims(Mel, 0) : Mel;
  debugArray(MelArray, "detectLanguage MelBatch");
  // Skip encoder forward pass if already-encoded audio features were given
  if (MelArray.shape(-2) != Model->Dims.NAudioCtx || 
      MelArray.shape(-1) != Model->Dims.NAudioState) {
    MelArray = Model->embedAudio(MelArray);
  }
  debugArray(MelArray, "detectLanguage AudioFeatures");
  // Forward pass using a single token, start of transcript
  int NAudio = MelArray.shape(0);
  std::vector<std::vector<int>> TokensVec(NAudio, {Tokenizer->getSot()});
  mx::array Tokens = mx::array(TokensVec[0].data(), {NAudio, 1}, mx::int32);
  
  mx::array Logits = Model->logits(Tokens, MelArray);
  Logits = mx::take(Logits, mx::array({0}), 1); // [:, 0]
  debugArray(Logits, "detectLanguage Logits");
  // Collect detected languages; suppress all non-language tokens
  mx::array MaskArray = mx::full({Logits.shape(-1)}, -std::numeric_limits<float>::infinity(), mx::float32);
  auto LangTokens = Tokenizer->getAllLanguageTokens();
  mx::array LangTokensArray = mx::array(LangTokens.data(), {static_cast<int>(LangTokens.size())}, mx::int32);
  debugArray(LangTokensArray, "LangTokensArray");
  debugArray(mx::zeros({static_cast<int>(LangTokens.size())}), "const std::string &Name);");
  MaskArray = mx::scatter(MaskArray, LangTokensArray, mx::zeros({static_cast<int>(LangTokens.size()), 1}), 0);
  
  Logits = Logits + MaskArray;
  mx::array LanguageTokens = mx::argmax(Logits, -1);
  mx::array LanguageTokenProbs = mx::softmax(Logits, -1);
  debugArray(LanguageTokens, "detectLanguage LanguageTokens");
  debugArray(LanguageTokenProbs, "detectLanguage LanguageTokenProbs");
  LanguageTokenProbs = mx::take(LanguageTokenProbs, 0, 0);
  
  // LanguageTokenProbs = mx::take(LanguageTokenProbs, 0);
  std::vector<std::map<std::string, float>> LanguageProbs;
  auto LangCodes = Tokenizer->getAllLanguageCodes();
  
  for (int I = 0; I < NAudio; ++I) {
    std::map<std::string, float> Probs;
    for (size_t J = 0; J < LangTokens.size() && J < LangCodes.size(); ++J) {
      mx::array ProbArray = mx::take(mx::take(LanguageTokenProbs, I, 0), LangTokens[J], 0);
      float Prob = ProbArray.item<float>();
      // std::cout << "Prob for " << LangCodes[J] << ": " << Prob << " " << I << " " << LangTokens[J] << std::endl;
      Probs[LangCodes[J]] = Prob;
    }
    LanguageProbs.push_back(Probs);
  }
  
  if (Single) {
    LanguageTokens = mx::take(LanguageTokens, mx::array({0}), 0);
    LanguageProbs = {LanguageProbs[0]};
  }
  
  return {LanguageTokens, LanguageProbs};
}

// Inference class implementation
Inference::Inference(std::shared_ptr<Whisper> Model) : Model(Model) {
  reset();
}

mx::array Inference::logits(const mx::array &Tokens, const mx::array &AudioFeatures) {
  // Fix the tuple unpacking issue - use only 2 elements
  auto [LogitsOutput, NewKvCache] = Model->forwardWithCrossQk(AudioFeatures, Tokens);
  // Convert vector<optional<array>> to proper cache format
  // TODO: Fix KV cache assignment when we know the exact type
  // KvCache = NewKvCache;
  return mx::astype(LogitsOutput, mx::float32);
}

void Inference::rearrangeKvCache(const std::vector<int> &SourceIndices) {
  if (!KvCache) return;
  // TODO: Implement KV cache rearrangement for beam search
}

void Inference::reset() {
  KvCache = std::nullopt;
}

// GreedyDecoder implementation
GreedyDecoder::GreedyDecoder(float Temperature, int Eot)
    : Temperature(Temperature), Eot(Eot) {}

void GreedyDecoder::reset() {
  // Nothing to reset for greedy decoder
}

std::tuple<mx::array, bool, mx::array> GreedyDecoder::update(
    const mx::array &Tokens, const mx::array &Logits, const mx::array &SumLogprobs) {
  
  int NBatch = Tokens.shape(0);
  
  // Sample next tokens
  mx::array NextTokens = mx::array({});
  if (Temperature == 0.0f) {
    NextTokens = mx::argmax(Logits, -1);
  } else {
    NextTokens = mx::random::categorical(Logits / Temperature);
  }
  
  // Compute logprobs
  mx::array Logprobs = Logits - mx::logsumexp(Logits, -1, true);
  mx::array CurrentLogprobs = mx::take_along_axis(Logprobs, mx::expand_dims(NextTokens, -1), -1);
  CurrentLogprobs = mx::squeeze(CurrentLogprobs, -1);
  
  // Check for EOT
  mx::array EotMask = mx::equal(mx::take(Tokens, mx::array({-1}), 1), mx::array({Eot}));
  EotMask = mx::squeeze(EotMask, -1);
  
  // Update tokens to set EOT for completed sequences
  NextTokens = NextTokens * (1 - mx::astype(EotMask, mx::int32)) + Eot * mx::astype(EotMask, mx::int32);
  
  // Update sum_logprobs, but don't add to sequences that have already ended
  mx::array NewSumLogprobs = SumLogprobs + CurrentLogprobs * (1.0f - mx::astype(EotMask, mx::float32));
  
  // Extend tokens
  mx::array NewTokens = mx::concatenate({Tokens, mx::expand_dims(NextTokens, -1)}, -1);
  
  // Check if all sequences are complete
  bool Completed = mx::all(mx::equal(mx::take(NewTokens, mx::array({-1}), 1), mx::array({Eot}))).item<bool>();
  
  return {NewTokens, Completed, NewSumLogprobs};
}

std::pair<mx::array, mx::array> GreedyDecoder::finalize(
    const mx::array &Tokens, const mx::array &SumLogprobs) {
  
  // Make sure each sequence has at least one EOT token at the end
  std::vector<std::pair<int, int>> PadWidths = {{0, 0}, {0, 1}};
  mx::array PaddedTokens = mx::pad(Tokens, PadWidths, mx::array({Eot}));
  
  return {PaddedTokens, SumLogprobs};
}

// SuppressBlank implementation
SuppressBlank::SuppressBlank(std::shared_ptr<::whisper::Tokenizer> Tokenizer, int SampleBegin, int NVocab)
    : SampleBegin(SampleBegin), Mask(mx::zeros({NVocab}, mx::float32)) {
  
  std::vector<float> MaskVec(NVocab, 0.0f);
  
  // Suppress space and EOT tokens
  auto SpaceTokens = Tokenizer->encode(" ");
  for (int Token : SpaceTokens) {
    if (Token >= 0 && Token < NVocab) {
      MaskVec[Token] = -std::numeric_limits<float>::infinity();
    }
  }
  
  int EotToken = Tokenizer->getEot();
  if (EotToken < NVocab) {
    MaskVec[EotToken] = -std::numeric_limits<float>::infinity();
  }
  
  Mask = mx::array(MaskVec.data(), {NVocab}, mx::float32);
}

mx::array SuppressBlank::apply(const mx::array &Logits, const mx::array &Tokens) {
  if (Tokens.shape(1) == SampleBegin) {
    return Logits + Mask;
  }
  return Logits;
}

// SuppressTokens implementation
SuppressTokens::SuppressTokens(const std::vector<int> &SuppressTokens, int NVocab) 
    : Mask(mx::zeros({NVocab}, mx::float32)) {
  
  std::vector<float> MaskVec(NVocab, 0.0f);
  for (int Token : SuppressTokens) {
    if (Token >= 0 && Token < NVocab) {
      MaskVec[Token] = -std::numeric_limits<float>::infinity();
    }
  }
  
  Mask = mx::array(MaskVec.data(), {NVocab}, mx::float32);
}

mx::array SuppressTokens::apply(const mx::array &Logits, const mx::array &Tokens) {
  return Logits + Mask;
}

// ApplyTimestampRules implementation
ApplyTimestampRules::ApplyTimestampRules(std::shared_ptr<::whisper::Tokenizer> Tokenizer, int SampleBegin,
                                       std::optional<int> MaxInitialTimestampIndex)
    : Tokenizer(Tokenizer), SampleBegin(SampleBegin), 
      MaxInitialTimestampIndex(MaxInitialTimestampIndex) {}

mx::array ApplyTimestampRules::apply(const mx::array &Logits, const mx::array &Tokens) {
  auto LogitsShape = Logits.shape();
  std::vector<float> MaskVec(LogitsShape[0] * LogitsShape[1], 0.0f);
  
  for (int K = 0; K < LogitsShape[0]; ++K) {
    // Convert tokens to vector for processing
    std::vector<int> Sequence;
    for (int I = SampleBegin; I < Tokens.shape(1); ++I) {
      // Use proper array access with mx::take
      mx::array TokenArray = mx::take(Tokens, mx::array({K, I}), 0);
      int Token = TokenArray.item<int>();
      if (Token == Tokenizer->getEot()) break;
      Sequence.push_back(Token);
    }
    
    bool LastWasTimestamp = !Sequence.empty() && Sequence.back() >= Tokenizer->getTimestampBegin();
    bool PenultimateWasTimestamp = Sequence.size() < 2 || Sequence[Sequence.size()-2] >= Tokenizer->getTimestampBegin();
    
    if (Tokens.shape(1) == SampleBegin) {
      // Suppress generating non-timestamp tokens at the beginning
      for (int I = 0; I < Tokenizer->getTimestampBegin(); ++I) {
        MaskVec[K * LogitsShape[1] + I] = -std::numeric_limits<float>::infinity();
      }
      
      if (MaxInitialTimestampIndex) {
        int LastAllowed = Tokenizer->getTimestampBegin() + *MaxInitialTimestampIndex;
        for (int I = LastAllowed + 1; I < LogitsShape[1]; ++I) {
          MaskVec[K * LogitsShape[1] + I] = -std::numeric_limits<float>::infinity();
        }
      }
    }
    
    if (LastWasTimestamp && PenultimateWasTimestamp) {
      // Cannot be normal text tokens
      for (int I = 0; I < Tokenizer->getEot(); ++I) {
        MaskVec[K * LogitsShape[1] + I] = -std::numeric_limits<float>::infinity();
      }
    }
    
    if (Tokenizer->getNoTimestamps()) {
      for (int I = 0; I < LogitsShape[0]; ++I) {
        MaskVec[I * LogitsShape[1] + Tokenizer->getNoTimestamps()] = -std::numeric_limits<float>::infinity();
      }
    }
  }
  
  mx::array MaskArray = mx::array(MaskVec.data(), LogitsShape, mx::float32);
  return Logits + MaskArray;
}

// MaximumLikelihoodRanker implementation
MaximumLikelihoodRanker::MaximumLikelihoodRanker(std::optional<float> LengthPenalty)
    : LengthPenalty(LengthPenalty) {}

std::vector<int> MaximumLikelihoodRanker::rank(
    const std::vector<std::vector<std::vector<int>>> &Tokens,
    const std::vector<std::vector<float>> &SumLogprobs) {
  
  std::vector<int> Selected;
  
  for (size_t I = 0; I < Tokens.size(); ++I) {
    std::vector<float> Scores;
    
    for (size_t J = 0; J < Tokens[I].size(); ++J) {
      int Length = Tokens[I][J].size();
      float Logprob = SumLogprobs[I][J];
      
      float Penalty;
      if (LengthPenalty) {
        Penalty = std::pow(Length, *LengthPenalty);
      } else {
        Penalty = Length;
      }
      
      Scores.push_back(Logprob / Penalty);
    }
    
    auto MaxIterator = std::max_element(Scores.begin(), Scores.end());
    Selected.push_back(std::distance(Scores.begin(), MaxIterator));
  }
  
  return Selected;
}

// DecodingTask implementation - Constructor and helper methods
DecodingTask::DecodingTask(std::shared_ptr<Whisper> Model, const DecodingOptions &Options)
    : Model(Model), Options(verifyOptions(Options)) {
  
  // Initialize tokenizer - equivalent to Python's get_tokenizer() call
  std::string Language = Options.Language.value_or("en");
  Tokenizer = whisper::getTokenizer(
      Model->isMultilingual(), 
      Model->numLanguages(),
      Language,
      Options.Task
  );
  
  NGroup = Options.BeamSize.value_or(Options.BestOf.value_or(1));
  NCtx = Model->Dims.NTextCtx;
  SampleLen = Options.SampleLen.value_or(NCtx / 2);
  
  // Handle SOT sequence with without_timestamps logic
  SotSequence = Tokenizer->SotSequence;
  if (Options.WithoutTimestamps) {
    SotSequence = Tokenizer->getSotSequenceIncludingNotimestamps();
  }
  
  InitialTokens = getInitialTokens();
  SampleBegin = InitialTokens.size();
  
  // Find SOT index
  auto SotToken = Tokenizer->getSot();
  auto Iterator = std::find(InitialTokens.begin(), InitialTokens.end(), SotToken);
  SotIndex = std::distance(InitialTokens.begin(), Iterator);
  
  // Initialize components
  Inference = std::make_unique<::whisper::Inference>(Model);
  SequenceRanker = std::make_unique<MaximumLikelihoodRanker>(Options.LengthPenalty);
  
  if (Options.BeamSize && *Options.BeamSize > 1) {
    throw std::runtime_error("Beam search decoder is not yet implemented");
  }
  Decoder = std::make_unique<GreedyDecoder>(Options.Temperature, Tokenizer->getEot());
  
  // Initialize logit filters in the same order as Python
  LogitFilters.clear(); // Make sure we start clean
  
  if (Options.SuppressBlank) {
    LogitFilters.push_back(std::make_unique<SuppressBlank>(Tokenizer, SampleBegin, Model->Dims.NVocab));
  }
  
  if (Options.SuppressTokens) {
    auto SuppressTokens = getSuppressTokens();
    LogitFilters.push_back(std::make_unique<::whisper::SuppressTokens>(SuppressTokens, Model->Dims.NVocab));
  }
  
  if (!Options.WithoutTimestamps) {
    std::optional<int> MaxInitialTimestampIndex;
    if (Options.MaxInitialTimestamp) {
      float Precision = 30.0f / Model->Dims.NAudioCtx; // CHUNK_LENGTH / n_audio_ctx
      MaxInitialTimestampIndex = static_cast<int>(std::round(*Options.MaxInitialTimestamp / Precision));
    }
    LogitFilters.push_back(std::make_unique<ApplyTimestampRules>(Tokenizer, SampleBegin, MaxInitialTimestampIndex));
  }
}

DecodingOptions DecodingTask::verifyOptions(const DecodingOptions &Options) {
  DecodingOptions Result = Options;
  
  // Check beam_size and best_of conflicts
  if (Result.BeamSize && Result.BestOf) {
    throw std::runtime_error("beam_size and best_of can't be given together");
  }
  
  // Check temperature = 0 with best_of
  if (Result.Temperature == 0.0f && Result.BestOf) {
    throw std::runtime_error("best_of with greedy sampling (T=0) is not compatible");
  }
  
  // Check patience requires beam_size
  if (Result.Patience && !Result.BeamSize) {
    throw std::runtime_error("patience requires beam_size to be given");
  }
  
  // Check length_penalty range
  if (Result.LengthPenalty && (*Result.LengthPenalty < 0.0f || *Result.LengthPenalty > 1.0f)) {
    throw std::runtime_error("length_penalty (alpha) should be a value between 0 and 1");
  }
  
  return Result;
}

std::vector<int> DecodingTask::getInitialTokens() {
  std::vector<int> Tokens = SotSequence;
  
  // Handle prefix first (like Python)
  if (Options.Prefix) {
    std::vector<int> PrefixTokens;
    if (std::holds_alternative<std::string>(*Options.Prefix)) {
      std::string PrefixStr = std::get<std::string>(*Options.Prefix);
      // Add space prefix like Python: " " + prefix.strip()
      PrefixTokens = Tokenizer->encode(" " + PrefixStr);
    } else {
      PrefixTokens = std::get<std::vector<int>>(*Options.Prefix);
    }
    
    if (SampleLen > 0) {
      int MaxPrefixLen = NCtx / 2 - SampleLen;
      if (static_cast<int>(PrefixTokens.size()) > MaxPrefixLen) {
        // Take last MaxPrefixLen tokens like Python: prefix_tokens[-max_prefix_len:]
        PrefixTokens = std::vector<int>(PrefixTokens.end() - MaxPrefixLen, PrefixTokens.end());
      }
    }
    
    // Append prefix tokens: tokens = tokens + prefix_tokens
    Tokens.insert(Tokens.end(), PrefixTokens.begin(), PrefixTokens.end());
  }
  
  // Handle prompt last (like Python)
  if (Options.Prompt) {
    std::vector<int> PromptTokens;
    if (std::holds_alternative<std::string>(*Options.Prompt)) {
      std::string PromptStr = std::get<std::string>(*Options.Prompt);
      // Add space prefix like Python: " " + prompt.strip()
      PromptTokens = Tokenizer->encode(" " + PromptStr);
    } else {
      PromptTokens = std::get<std::vector<int>>(*Options.Prompt);
    }
    
    int MaxPromptLen = NCtx / 2 - 1;
    if (static_cast<int>(PromptTokens.size()) > MaxPromptLen) {
      // Take last MaxPromptLen tokens like Python: prompt_tokens[-(n_ctx // 2 - 1):]
      PromptTokens = std::vector<int>(PromptTokens.end() - MaxPromptLen, PromptTokens.end());
    }
    
    // Prepend sot_prev and prompt tokens, then append original tokens
    // Python: [tokenizer.sot_prev] + prompt_tokens + tokens
    std::vector<int> NewTokens;
    NewTokens.push_back(Tokenizer->getSotPrev());
    NewTokens.insert(NewTokens.end(), PromptTokens.begin(), PromptTokens.end());
    NewTokens.insert(NewTokens.end(), Tokens.begin(), Tokens.end());
    Tokens = NewTokens;
  }
  
  return Tokens;
}

std::vector<int> DecodingTask::getSuppressTokens() {
  std::vector<int> SuppressTokens;
  
  if (Options.SuppressTokens) {
    if (std::holds_alternative<std::string>(*Options.SuppressTokens)) {
      std::string TokensString = std::get<std::string>(*Options.SuppressTokens);
      // Split by comma like Python version
      std::istringstream Iss(TokensString);
      std::string TokenString;
      while (std::getline(Iss, TokenString, ',')) {
        if (!TokenString.empty()) {
          SuppressTokens.push_back(std::stoi(TokenString));
        }
      }
    } else {
      SuppressTokens = std::get<std::vector<int>>(*Options.SuppressTokens);
    }
  }
  
  // Handle -1 (non-speech tokens) - same logic as Python
  auto Iterator = std::find(SuppressTokens.begin(), SuppressTokens.end(), -1);
  if (Iterator != SuppressTokens.end()) {
    SuppressTokens.erase(Iterator);
    auto NonSpeechTokens = Tokenizer->getNonSpeechTokens();
    SuppressTokens.insert(SuppressTokens.end(), NonSpeechTokens.begin(), NonSpeechTokens.end());
  } else if (SuppressTokens.empty()) {
    // Python: elif suppress_tokens is None or len(suppress_tokens) == 0: suppress_tokens = []
    // Already empty, nothing to do
  }
  
  // Add standard suppress tokens like Python version
  SuppressTokens.push_back(Tokenizer->getTranscribe());
  SuppressTokens.push_back(Tokenizer->getTranslate());
  SuppressTokens.push_back(Tokenizer->getSot());
  SuppressTokens.push_back(Tokenizer->getSotPrev());
  SuppressTokens.push_back(Tokenizer->getSotLm());
  
  // Add no_speech token if it exists
  if (Tokenizer->getNoSpeech() != -1) { // Assuming -1 means not available
    SuppressTokens.push_back(Tokenizer->getNoSpeech());
  }
  
  // Remove duplicates and sort like Python: sorted(set(suppress_tokens))
  std::sort(SuppressTokens.begin(), SuppressTokens.end());
  SuppressTokens.erase(std::unique(SuppressTokens.begin(), SuppressTokens.end()), SuppressTokens.end());
  
  return SuppressTokens;
}

mx::array DecodingTask::getAudioFeatures(const mx::array &Mel) {
  bool Single = Mel.ndim() == 2;
  mx::array MelArray = Single ? mx::expand_dims(Mel, 0) : Mel;
  
  mx::array AudioFeatures = MelArray;
  
  // Skip encoder forward pass if already-encoded audio features were given
  if (AudioFeatures.shape(-2) != Model->Dims.NAudioCtx || 
      AudioFeatures.shape(-1) != Model->Dims.NAudioState) {
    AudioFeatures = Model->embedAudio(AudioFeatures);
  }
  
  return AudioFeatures;
}

std::pair<std::vector<std::string>, std::optional<std::vector<std::map<std::string, float>>>>
DecodingTask::detectLanguage(const mx::array &AudioFeatures, mx::array &Tokens) {
  
  std::vector<std::string> Languages(AudioFeatures.shape(0), Options.Language.value_or("en"));
  std::optional<std::vector<std::map<std::string, float>>> LangProbs;
  
  if (Options.Task == "lang_id") {
    if (Tokenizer->getNoSpeech()) {
      // TODO: Implement no speech probability computation
      mx::array NoSpeechProbs = mx::zeros({AudioFeatures.shape(0)}, mx::float32);
    }
    
    // Write language tokens to the tokens array
    auto [detectedLanguages, probabilities] = ::whisper::detectLanguage(Model, AudioFeatures, Tokenizer);
    
    Languages.clear();
    for (const auto &ProbsMap : probabilities) {
      auto MaxIterator = std::max_element(ProbsMap.begin(), ProbsMap.end(),
          [](const auto &A, const auto &B) { return A.second < B.second; });
      Languages.push_back(MaxIterator->first);
    }
    
    LangProbs = probabilities;
  }
  
  return {Languages, LangProbs};
}

std::tuple<mx::array, mx::array, mx::array> DecodingTask::mainLoop(
    const mx::array &AudioFeatures, const mx::array &Tokens) {
  
  int NAudio = AudioFeatures.shape(0);
  
  Inference->reset();
  Decoder->reset();
  
  mx::array CurrentTokens = Tokens;
  mx::array SumLogprobs = mx::zeros({NAudio}, mx::float32);
  
  // Continue generation
  for (int I = 0; I < SampleLen; ++I) {
    if (CurrentTokens.shape(-1) > NCtx) break;
    
    auto StepFunction = [&](const mx::array &Inputs, const mx::array &AudioFeats,
                  const mx::array &TokSeq, const mx::array &SumLogp) {
      mx::array PreLogits = Inference->logits(Inputs, AudioFeats);
      mx::array Logits = mx::take(PreLogits, mx::array({-1}), 1); // [:, -1]
      
      // Apply logit filters
      for (const auto &Filter : LogitFilters) {
        Logits = Filter->apply(Logits, TokSeq);
      }
      
      return Decoder->update(TokSeq, Logits, SumLogp);
    };
    
    mx::array Inputs = mx::take(CurrentTokens, mx::array({-1}), -1);
    auto [NextTokens, NextCompleted, NextSumLogprobs] =
        StepFunction(Inputs, AudioFeatures, CurrentTokens, SumLogprobs);
    
    CurrentTokens = NextTokens;
    SumLogprobs = NextSumLogprobs;
    
    if (NextCompleted) break;
  }
  
  auto [FinalizedTokens, FinalizedLogprobs] = Decoder->finalize(CurrentTokens, SumLogprobs);
  
  // Compute no_speech_probs
  mx::array NoSpeechProbs = mx::zeros({NAudio}, mx::float32);
  
  if (Tokenizer->getNoSpeech()) {
    // TODO: Implement proper no speech probability computation
    NoSpeechProbs = mx::zeros({NAudio}, mx::float32);
  }
  
  return {FinalizedTokens, FinalizedLogprobs, NoSpeechProbs};
}

std::vector<DecodingResult> DecodingTask::run(const mx::array &Mel) {
  mx::array AudioFeatures = getAudioFeatures(Mel);
  int NAudio = AudioFeatures.shape(0);
  
  // Prepare initial tokens
  mx::array Tokens = mx::array(InitialTokens.data(), {static_cast<int>(InitialTokens.size())}, mx::int32);
  Tokens = mx::broadcast_to(Tokens, {NAudio, static_cast<int>(InitialTokens.size())});
  
  // Language detection
  auto [Languages, LangProbs] = detectLanguage(AudioFeatures, Tokens);
  
  // Main decoding loop
  auto [FinalizedTokens, FinalizedLogprobs, NoSpeechProbs] = mainLoop(AudioFeatures, Tokens);
  
  // Convert to results
  std::vector<DecodingResult> Results;
  
  for (int I = 0; I < NAudio; ++I) {
    DecodingResult Result;
    Result.AudioFeatures = mx::take(AudioFeatures, mx::array({I}), 0);
    Result.Language = Languages[I];
    
    if (LangProbs) {
      Result.LanguageProbs = (*LangProbs)[I];
    }
    
    // Extract tokens for this sequence
    std::vector<int> SeqTokens;
    for (int K = 0; K < FinalizedTokens.shape(-1); ++K) {
      // Use proper array access with mx::take
      mx::array TokenArray = mx::take(FinalizedTokens, mx::array({I, K}), 0);
      int Token = TokenArray.item<int>();
      if (Token == Tokenizer->getEot()) break;
      SeqTokens.push_back(Token);
    }
    
    Result.Tokens = SeqTokens;
    Result.Text = Tokenizer ? Tokenizer->decode(SeqTokens) : "";
    
    // Use proper array access for scalar values  
    mx::array AvgLogprobArray = mx::take(FinalizedLogprobs, mx::array({I}), 0);
    Result.AvgLogprob = AvgLogprobArray.item<float>();
    
    mx::array NoSpeechArray = mx::take(NoSpeechProbs, mx::array({I}), 0);
    Result.NoSpeechProb = NoSpeechArray.item<float>();
    Result.Temperature = Options.Temperature;
    Result.CompressionRatio = compressionRatio(Result.Text);
    
    Results.push_back(Result);
  }
  
  return Results;
}

// Main decode function
std::variant<DecodingResult, std::vector<DecodingResult>>
decode(std::shared_ptr<Whisper> Model, const mx::array &Mel,
       const DecodingOptions &Options) {
  
  auto Results = DecodingTask(Model, Options).run(Mel);
  
  if (Results.size() == 1) {
    return Results[0];
  }
  return Results;
}

} // namespace whisper
