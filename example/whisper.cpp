#include "model/whisper/whisper.h"
#include "model/whisper_transcribe.h"

int main() {
  std::string ModelPath = "../whisper-tiny";
  auto Whisper = whisper::Whisper::fromPretrained(ModelPath);

  std::string AudioPath = "../example/audio.mp3";
  auto Result = whisper::transcribe(AudioPath, Whisper, true);
  std::cout << Result.Text << std::endl;
  return 0;
}