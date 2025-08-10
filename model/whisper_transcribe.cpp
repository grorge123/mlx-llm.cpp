#include "whisper_transcribe.h"
#include "base.h"
#include "spdlog/spdlog.h"
#include "whisper/tokenizer.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mlx/array.h>
#include <mlx/fft.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>
#include <sstream>
#include <vector>
#include <zlib.h>

namespace whisper {

// Remove duplicate LANGUAGES definition since it's now in tokenizer.h
// Remove duplicate LANGUAGES definition since it's now in tokenizer.h

mx::array loadAudio(const std::string &FilePath, int SampleRate) {
  int Channels = 1;
  std::vector<std::string> Cmd = {"ffmpeg",   "-nostdin",
                                  "-threads", "0",
                                  "-i",       FilePath,
                                  "-f",       "s16le",
                                  "-ac",      std::to_string(Channels),
                                  "-acodec",  "pcm_s16le",
                                  "-ar",      std::to_string(SampleRate),
                                  "-v",       "quiet",
                                  "-"};

  // Build command string
  std::stringstream CmdStream;
  for (size_t i = 0; i < Cmd.size(); ++i) {
    if (i > 0)
      CmdStream << " ";
    CmdStream << Cmd[i];
  }
  std::string CmdString = CmdStream.str();

  // Execute ffmpeg command
  FILE *Pipe = popen(CmdString.c_str(), "r");
  if (!Pipe) {
    throw std::runtime_error("Failed to execute ffmpeg command");
  }

  // Read raw audio data
  std::vector<int16_t> AudioData;
  int16_t Buffer[4096];
  size_t BytesRead;

  while ((BytesRead = fread(Buffer, sizeof(int16_t), 4096, Pipe)) > 0) {
    AudioData.insert(AudioData.end(), Buffer, Buffer + BytesRead);
  }

  int ExitCode = pclose(Pipe);
  if (ExitCode != 0) {
    throw std::runtime_error("ffmpeg command failed with exit code: " +
                             std::to_string(ExitCode));
  }

  if (AudioData.empty()) {
    throw std::runtime_error("No audio data loaded from file: " + FilePath);
  }

  mx::array AudioArray = mx::array(
      AudioData.data(), {static_cast<int>(AudioData.size())}, mx::int16);

  if (Channels > 1) {
    int Frames = AudioData.size() / Channels;
    AudioArray = mx::reshape(AudioArray, {Frames, Channels});
  }

  // Convert to float32 and normalize by 32768.0 (not 32767.0 like Python)
  mx::array FloatAudio = mx::astype(AudioArray, mx::float32) / 32768.0f;

  return FloatAudio;
}

mx::array padOrTrim(const mx::array &Array, int Length, int Axis) {
  auto Shape = Array.shape();
  int ActualAxis = Axis < 0 ? Shape.size() + Axis : Axis;

  if (Shape[ActualAxis] > Length) {
    std::vector<int> Start(Shape.size(), 0);
    std::vector<int> End = Shape;
    std::vector<int> Strides(Shape.size(), 1);
    End[ActualAxis] = Length;
    return mx::slice(Array, Start, End, Strides);
  }

  if (Shape[ActualAxis] < Length) {
    std::vector<std::pair<int, int>> PadWidths(Shape.size(), {0, 0});
    PadWidths[ActualAxis] = {0, Length - Shape[ActualAxis]};
    return mx::pad(Array, PadWidths);
  }

  return Array;
}

mx::array hanningWindow(int Size) {
  mx::array N = mx::arange(Size + 1);
  mx::array Window = 0.5f * (1.0f - mx::cos(2.0f * M_PI * N / Size));
  std::vector<int> Start = {0};
  std::vector<int> End = {-1};
  std::vector<int> Strides = {1};
  return mx::slice(Window, Start, End, Strides);
}

mx::array stft(const mx::array &X, const mx::array &Window, int NPerseg = 256,
               int NOverlap = 0, int NfFt = 0,
               const std::string &PadMode = "reflect") {
  if (NfFt == 0)
    NfFt = NPerseg;

  auto Pad = [](const mx::array &X, int Padding, const std::string &PadMode) {
    if (PadMode == "constant") {
      std::vector<std::pair<int, int>> PadWidths = {{Padding, Padding}};
      return mx::pad(X, PadWidths);
    }
    if (PadMode == "reflect") {
      // Reflect padding like Python: prefix = x[1:padding+1][::-1]
      std::vector<int> Start(X.shape().size(), 0);
      std::vector<int> End = X.shape();
      std::vector<int> Strides(X.shape().size(), 1);

      Start[0] = 1;
      End[0] = Padding + 1;
      auto Prefix = mx::slice(X, Start, End, Strides);
      Start[0] = -(Padding + 1);
      End[0] = -1;
      auto Suffix = mx::slice(X, Start, End, Strides);

      return mx::concatenate({Prefix, X, Suffix});
    }
    throw std::runtime_error("Invalid pad mode: " + PadMode);
  };

  // Pad the signal
  int Padding = NPerseg / 2;
  auto Padded = Pad(X, Padding, PadMode);
  std::vector<int64_t> Strides = {static_cast<int64_t>(NOverlap), 1};
  int T = (Padded.shape(0) - NPerseg + NOverlap) / NOverlap;
  auto Shape = std::vector<int>{T, NfFt};
  auto StridedX = mx::as_strided(Padded, Shape, Strides, 0);

  return mx::fft::rfft(StridedX * Window);
}

mx::array melFilters(int NMels) {
  // Load precomputed mel filters from file
  if (NMels != 80 && NMels != 128) {
    spdlog::error("Unsupported number of mel filters: " +
                  std::to_string(NMels));
    assumingUnreachable();
  }
  std::string FileName = "assets/mel_filters_" + std::to_string(NMels) + ".npy";
  return mx::load(FileName);
}

mx::array logMelSpectrogram(const mx::array &Audio, int NMels, int Padding) {
  auto PaddedAudio = Audio;
  if (Padding > 0) {
    std::vector<std::pair<int, int>> PadWidths = {{0, Padding}};
    PaddedAudio = mx::pad(Audio, PadWidths);
  }

  auto Window = hanningWindow(DefaultNFft);
  auto Freqs = stft(PaddedAudio, Window, DefaultNFft, DefaultHopLength,
                    DefaultNFft, "reflect");
  // Get magnitudes and square them - match Python: freqs[:-1, :].abs().square()
  std::vector<int> Start = {0, 0};
  std::vector<int> End = {static_cast<int>(Freqs.shape(0) - 1),
                          static_cast<int>(Freqs.shape(1))};
  std::vector<int> Strides = {1, 1};
  auto FreqsSliced = mx::slice(Freqs, Start, End, Strides);
  auto Magnitudes = mx::square(mx::abs(FreqsSliced));
  // Apply mel filters
  auto Filters = melFilters(NMels);
  auto MelSpec = mx::matmul(Magnitudes, mx::transpose(Filters));
  // Convert to log scale - match Python exactly
  auto LogSpec =
      mx::log10(mx::maximum(MelSpec, mx::full(MelSpec.shape(), 1e-10f)));
  LogSpec = mx::maximum(LogSpec, mx::max(LogSpec) - 8.0f);
  LogSpec = (LogSpec + 4.0f) / 4.0f;
  return LogSpec;
}

// Utility functions
std::string formatTimestamp(float Seconds) {
  assert(Seconds >= 0);
  int Milliseconds = static_cast<int>(std::round(Seconds * 1000.0f));

  int Hours = Milliseconds / 3600000;
  Milliseconds -= Hours * 3600000;

  int Minutes = Milliseconds / 60000;
  Milliseconds -= Minutes * 60000;

  int Secs = Milliseconds / 1000;
  Milliseconds -= Secs * 1000;

  std::stringstream Ss;
  if (Hours > 0) {
    Ss << std::setfill('0') << std::setw(2) << Hours << ":";
  }
  Ss << std::setfill('0') << std::setw(2) << Minutes << ":" << std::setfill('0')
     << std::setw(2) << Secs << "." << std::setfill('0') << std::setw(3)
     << Milliseconds;

  return Ss.str();
}

std::optional<float> getEnd(const std::vector<TranscribeSegment> &Segments) {
  for (auto It = Segments.rbegin(); It != Segments.rend(); ++It) {
    for (auto WordIt = It->Words.rbegin(); WordIt != It->Words.rend();
         ++WordIt) {
      return WordIt->End;
    }
    if (!It->Words.empty()) {
      return It->End;
    }
  }
  return Segments.empty() ? std::nullopt
                          : std::make_optional(Segments.back().End);
}

// float compressionRatio(const std::string &Text) {
//   std::vector<uint8_t> Data(Text.begin(), Text.end());

//   // Simple compression ratio using zlib
//   z_stream Stream;
//   Stream.zalloc = Z_NULL;
//   Stream.zfree = Z_NULL;
//   Stream.opaque = Z_NULL;

//   if (deflateInit(&Stream, Z_DEFAULT_COMPRESSION) != Z_OK) {
//     return 1.0f;
//   }

//   std::vector<uint8_t> Compressed(Data.size() * 2);
//   Stream.next_in = Data.data();
//   Stream.avail_in = Data.size();
//   Stream.next_out = Compressed.data();
//   Stream.avail_out = Compressed.size();

//   if (deflate(&Stream, Z_FINISH) != Z_STREAM_END) {
//     deflateEnd(&Stream);
//     return 1.0f;
//   }

//   size_t CompressedSize = Compressed.size() - Stream.avail_out;
//   deflateEnd(&Stream);

//   return static_cast<float>(Data.size()) /
//   static_cast<float>(CompressedSize);
// }

// Language detection
// std::pair<mx::array, std::vector<std::map<std::string, float>>>
// detectLanguage(std::shared_ptr<whisper::Whisper> Model, const mx::array &Mel,
//                std::shared_ptr<whisper::Tokenizer> Tokenizer) {
//   if (!Tokenizer) {
//     Tokenizer = getTokenizer(Model->isMultilingual(), Model->numLanguages());
//   }
//   debugArray(Mel, "detectLanguage Mel");
//   bool Single = Mel.ndim() == 2;
//   mx::array MelBatch = Single ? mx::expand_dims(Mel, 0) : Mel;
//   debugArray(MelBatch, "detectLanguage MelBatch");
//   // Skip encoder if already encoded
//   mx::array AudioFeatures = (MelBatch.shape(-2) != Model->Dims.NAudioCtx ||
//                              MelBatch.shape(-1) != Model->Dims.NAudioState)
//                                 ? Model->embedAudio(MelBatch)
//                                 : MelBatch;
//   debugArray(AudioFeatures, "detectLanguage AudioFeatures");
//   // Forward pass with start of transcript token
//   int NAudio = AudioFeatures.shape(0);

//   // Get SOT token from tokenizer
//   int SotToken = Tokenizer->getSot();
//   auto X = mx::full({NAudio, 1}, SotToken, mx::int32);
//   auto Logits = Model->logits(X, AudioFeatures);
//   auto Indices = mx::array({0}, mx::int32);
//   Logits = mx::take(Logits, Indices, 1); // Take first token logits
//   debugArray(Logits, "detectLanguage Logits");

//   // Apply language token mask (simplified)
//   auto LanguageTokens = mx::argmax(Logits, -1);
//   auto LanguageProbs = mx::softmax(Logits, -1);

//   // Convert to output format
//   std::vector<std::map<std::string, float>> ResultProbs(NAudio);

//   if (Single) {
//     LanguageTokens = mx::take(LanguageTokens, mx::array({0}), 0);
//     ResultProbs.resize(1);
//   }

//   return {LanguageTokens, ResultProbs};
// }

// Decoding functions
DecodingResult
decodeWithFallback(std::shared_ptr<whisper::Whisper> Model,
                   const mx::array &MelSegment,
                   const DecodingOptions &DecodeOptions,
                   const std::vector<float> &Temperatures,
                   std::unique_ptr<whisper::Tokenizer> &Tokenizer,
                   std::optional<float> CompressionRatioThreshold,
                   std::optional<float> LogprobThreshold,
                   std::optional<float> NoSpeechThreshold) {

  DecodingResult Result;
  Result.AudioFeatures = MelSegment;
  Result.Language = "";
  Result.Tokens = std::vector<int32_t>();
  Result.Text = "";

  for (float Temp : Temperatures) {
    // Simple greedy decoding for temperature = 0, sampling otherwise
    DecodingOptions Options = DecodeOptions;
    Options.Temperature = Temp;

    // This is a simplified decode implementation
    // In practice, you'd implement the full beam search/sampling logic
    std::vector<int32_t> Tokens;

    // Start with SOT sequence
    auto SotSequence = Tokenizer->SotSequence;
    Tokens.insert(Tokens.end(), SotSequence.begin(), SotSequence.end());

    // Generate tokens
    mx::array CurrentTokens = mx::array(
        Tokens.data(), {1, static_cast<int>(Tokens.size())}, mx::int32);
    Result = std::get<DecodingResult>(decode(Model, MelSegment, Options));

    // Check fallback conditions
    bool NeedsFallback = false;
    if (CompressionRatioThreshold &&
        Result.CompressionRatio > *CompressionRatioThreshold) {
      NeedsFallback = true;
    }
    if (LogprobThreshold && Result.AvgLogprob < *LogprobThreshold) {
      NeedsFallback = true;
    }
    if (NoSpeechThreshold && Result.NoSpeechProb > *NoSpeechThreshold) {
      NeedsFallback = true;
    }

    if (!NeedsFallback) {
      break;
    }
  }

  return Result;
}

// Word-level timestamp functions
void addWordTimestamps(std::vector<TranscribeSegment> &Segments,
                       std::shared_ptr<whisper::Whisper> Model,
                       std::unique_ptr<whisper::Tokenizer> &Tokenizer,
                       const mx::array &Mel, int NumFrames,
                       const std::string &PrependPunctuations,
                       const std::string &AppendPunctuations,
                       float LastSpeechTimestamp) {

  // This is a simplified implementation
  // Full implementation would use cross-attention patterns and DTW
  for (auto &Segment : Segments) {
    if (!Segment.Tokens.empty()) {
      float Duration = Segment.End - Segment.Start;
      float TimePerToken = Duration / Segment.Tokens.size();

      // Simple uniform distribution of word timestamps
      std::string Text = Segment.Text;
      std::istringstream Iss(Text);
      std::string Word;
      int WordIdx = 0;

      while (Iss >> Word && WordIdx < Segment.Tokens.size()) {
        WordInfo WordInfo;
        WordInfo.Word = Word;
        WordInfo.Start = Segment.Start + WordIdx * TimePerToken;
        WordInfo.End = Segment.Start + (WordIdx + 1) * TimePerToken;
        WordInfo.Probability = 0.8f; // Default probability

        Segment.Words.push_back(WordInfo);
        WordIdx++;
      }
    }
  }
}

// Anomaly detection functions
float wordAnomalyScore(const WordInfo &Word) {
  float Score = 0.0f;

  if (Word.Probability < 0.15f) {
    Score += 1.0f;
  }

  float Duration = Word.End - Word.Start;
  if (Duration < 0.133f) {
    Score += (0.133f - Duration) * 15.0f;
  }
  if (Duration > 2.0f) {
    Score += Duration - 2.0f;
  }

  return Score;
}

bool isSegmentAnomaly(const std::optional<TranscribeSegment> &Segment) {
  if (!Segment || Segment->Words.empty()) {
    return false;
  }

  // Filter out punctuation words
  std::vector<WordInfo> FilteredWords;
  std::string Punctuation = "\"'?([{-\"'.,!?:\")]},";

  for (const auto &Word : Segment->Words) {
    if (Punctuation.find(Word.Word) == std::string::npos) {
      FilteredWords.push_back(Word);
    }
  }

  if (FilteredWords.size() > 8) {
    FilteredWords.resize(8);
  }

  float TotalScore = 0.0f;
  for (const auto &Word : FilteredWords) {
    TotalScore += wordAnomalyScore(Word);
  }

  return TotalScore >= 3.0f || TotalScore + 0.01f >= FilteredWords.size();
}

std::optional<TranscribeSegment>
nextWordsSegment(const std::vector<TranscribeSegment> &Segments) {
  for (const auto &Segment : Segments) {
    if (!Segment.Words.empty()) {
      return Segment;
    }
  }
  return std::nullopt;
}

// Main transcribe function
TranscribeResult
transcribe(const std::variant<std::string, mx::array> &Audio,
           std::shared_ptr<whisper::Whisper> Model, std::optional<bool> Verbose,
           std::variant<float, std::vector<float>> Temperature,
           std::optional<float> CompressionRatioThreshold,
           std::optional<float> LogprobThreshold,
           std::optional<float> NoSpeechThreshold, bool ConditionOnPreviousText,
           std::optional<std::string> InitialPrompt, bool WordTimestamps,
           const std::string &PrependPunctuations,
           const std::string &AppendPunctuations,
           std::variant<std::string, std::vector<float>> ClipTimestamps,
           std::optional<float> HallucinationSilenceThreshold,
           const DecodingOptions &DecodeOptions) {

  // Get audio array
  mx::array AudioArray =
      std::holds_alternative<std::string>(Audio)
          ? loadAudio(std::get<std::string>(Audio), DefaultSampleRate)
          : std::get<mx::array>(Audio);

  // Get dtype
  mx::Dtype Dtype = DecodeOptions.Fp16 ? mx::float16 : mx::float32;

  // Generate mel spectrogram
  auto Mel = logMelSpectrogram(AudioArray, Model->Dims.NMels, DefaultNSamples);
  int ContentFrames = Mel.shape(-2) - DefaultNFrames;
  float ContentDuration =
      static_cast<float>(ContentFrames * DefaultHopLength) / DefaultSampleRate;

  // Language detection
  DecodingOptions Options = DecodeOptions;
  if (!Options.Language) {
    if (!Model->isMultilingual()) {
      Options.Language = "en";
    } else {
      if (Verbose.value_or(false)) {
        std::cout << "Detecting language using up to the first 30 seconds.\n";
      }

      auto MelSegment = padOrTrim(Mel, DefaultNFrames, -2);
      MelSegment = mx::astype(MelSegment, Dtype);
      auto [LangTokens, LangProbs] = detectLanguage(Model, MelSegment);

      // Find most probable language from the detection results
      if (!LangProbs.empty() && !LangProbs[0].empty()) {
        // Find language with highest probability
        auto MaxIterator = std::max_element(
            LangProbs[0].begin(), LangProbs[0].end(),
            [](const auto &A, const auto &B) { return A.second < B.second; });
        Options.Language = MaxIterator->first;
      } else {
        Options.Language = "en"; // Default fallback only if detection failed
      }

      if (Verbose.value_or(false)) {
        std::string LangName = findLanguageByCode(*Options.Language);
        if (LangName.empty()) {
          LangName = *Options.Language; // fallback to code if not found
        }
        std::cout << "Detected language: " << LangName << std::endl;
      }
    }
  }

  auto Task = Options.Task;
  auto Tokenizer = getTokenizer(Model->isMultilingual(), Model->numLanguages(),
                                *Options.Language, Task);

  // Parse clip timestamps
  std::vector<float> ClipTimes;
  if (std::holds_alternative<std::string>(ClipTimestamps)) {
    std::string ClipStr = std::get<std::string>(ClipTimestamps);
    if (!ClipStr.empty()) {
      std::stringstream Ss(ClipStr);
      std::string Item;
      while (std::getline(Ss, Item, ',')) {
        ClipTimes.push_back(std::stof(Item));
      }
    }
  } else {
    ClipTimes = std::get<std::vector<float>>(ClipTimestamps);
  }

  // Set up seek points
  std::vector<int> SeekPoints;
  for (float Ts : ClipTimes) {
    SeekPoints.push_back(
        static_cast<int>(std::round(Ts * DefaultFramesPerSecond)));
  }
  if (SeekPoints.empty()) {
    SeekPoints.push_back(0);
  }
  if (SeekPoints.size() % 2 == 1) {
    SeekPoints.push_back(ContentFrames);
  } else {
    SeekPoints.back() = std::min(ContentFrames, SeekPoints.back());
  }

  // Create seek clips
  std::vector<std::pair<int, int>> SeekClips;
  for (size_t I = 0; I < SeekPoints.size(); I += 2) {
    SeekClips.emplace_back(SeekPoints[I], SeekPoints[I + 1]);
  }

  int Seek = SeekClips[0].first;
  // time_precision

  std::vector<int32_t> AllTokens;
  std::vector<TranscribeSegment> AllSegments;
  // prompt_reset_since

  if (InitialPrompt) {
    auto PromptTokens = Tokenizer->encode(" " + *InitialPrompt);
    AllTokens.insert(AllTokens.end(), PromptTokens.begin(), PromptTokens.end());
  }

  std::vector<float> Temperatures;
  if (std::holds_alternative<float>(Temperature)) {
    Temperatures = {std::get<float>(Temperature)};
  } else {
    Temperatures = std::get<std::vector<float>>(Temperature);
  }

  // Processing loop
  for (const auto &[SeekClipStart, SeekClipEnd] : SeekClips) {
    while (Seek < SeekClipEnd) {
      float TimeOffset =
          static_cast<float>(Seek * DefaultHopLength) / DefaultSampleRate;
      int SegmentSize =
          std::min({DefaultNFrames, ContentFrames - Seek, SeekClipEnd - Seek});

      // Extract mel segment
      std::vector<int> Start(Mel.shape().size(), 0);
      std::vector<int> End = Mel.shape();
      Start[0] = Seek;
      End[0] = Seek + SegmentSize;
      auto MelSegment = mx::slice(Mel, Start, End);
      MelSegment = padOrTrim(MelSegment, DefaultNFrames, -2);
      MelSegment = mx::astype(MelSegment, Dtype);

      // Decode segment
      auto Result = decodeWithFallback(Model, MelSegment, Options, Temperatures,
                                       Tokenizer, CompressionRatioThreshold,
                                       LogprobThreshold, NoSpeechThreshold);

      // Create segment
      TranscribeSegment Segment;
      Segment.Id = AllSegments.size();
      Segment.Seek = Seek;
      Segment.Start = TimeOffset;
      Segment.End =
          TimeOffset + static_cast<float>(SegmentSize * DefaultHopLength) /
                           DefaultSampleRate;
      Segment.Text = Result.Text;
      Segment.Tokens = Result.Tokens;
      Segment.Temperature = Result.Temperature;
      Segment.AvgLogprob = Result.AvgLogprob;
      Segment.CompressionRatio = Result.CompressionRatio;
      Segment.NoSpeechProb = Result.NoSpeechProb;

      // Add word timestamps if requested
      if (WordTimestamps) {
        std::vector<TranscribeSegment> TempSegments = {Segment};
        addWordTimestamps(TempSegments, Model, Tokenizer, MelSegment,
                          SegmentSize, PrependPunctuations, AppendPunctuations);
        Segment.Words = TempSegments[0].Words;
      }

      // Add to results
      AllSegments.push_back(Segment);
      AllTokens.insert(AllTokens.end(), Result.Tokens.begin(),
                       Result.Tokens.end());

      // Update seek position
      Seek += SegmentSize;

      // Verbose output
      if (Verbose.value_or(false)) {
        std::cout << "[" << formatTimestamp(Segment.Start) << " --> "
                  << formatTimestamp(Segment.End) << "] " << Segment.Text
                  << std::endl;
      }
    }
  }

  // Prepare final result
  TranscribeResult FinalResult;
  FinalResult.Segments = AllSegments;
  FinalResult.Language = *Options.Language;

  // Concatenate all text
  std::stringstream TextStream;
  for (const auto &Segment : AllSegments) {
    TextStream << Segment.Text;
  }
  FinalResult.Text = TextStream.str();

  return FinalResult;
}

} // namespace whisper
