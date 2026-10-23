import Foundation
import MLX
import MLXAudioCore
import MLXAudioSTT
import Testing

/// CI-only: does the Swift loader read mlx-audio's q8/q4 conversions of the multilingual
/// Granite Speech 5.0 TurboCTC (IBM revision 1dd23f4), and what do they transcribe?
struct GraniteMultilingualQuantCheck {
    @Test func transcribeClips() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let q8 = env["GRANITE_Q8_DIR"], let q4 = env["GRANITE_Q4_DIR"] else { return }
        let media = Bundle.module.resourceURL!.appendingPathComponent("media/granite")
        let clips = try FileManager.default.contentsOfDirectory(at: media, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "wav" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        let variants: [(String, () async throws -> GraniteSpeech5CTCModel)] = [
            ("bf16", { try await GraniteSpeech5CTCModel.fromPretrained("ibm-granite/granite-speech-5.0-470m-turboctc") }),
            ("q8", { try GraniteSpeech5CTCModel.fromModelDirectory(URL(fileURLWithPath: q8)) }),
            ("q4", { try GraniteSpeech5CTCModel.fromModelDirectory(URL(fileURLWithPath: q4)) }),
        ]
        for (name, load) in variants {
            do {
                let model = try await load()
                for clip in clips {
                    let (_, audio) = try loadAudioArray(from: clip, sampleRate: 16000)
                    let out = model.generate(audio: audio, generationParameters: model.defaultGenerationParameters)
                    print("GRANITE \(name) \(clip.deletingPathExtension().lastPathComponent) | \(out.text)")
                }
            } catch {
                print("GRANITE \(name) LOAD_FAILED | \(error)")
            }
        }
    }
}
