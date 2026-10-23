import Foundation
import MLXAudioCore
import MLXAudioSTT
import Testing

/// CI-only benchmark for Blaizzy#249: one 20-minute LibriSpeech recording, several chunk lengths.
struct Qwen3ASRChunkBench {
    @Test func chunkDurationBench() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_BENCH_249"] == "1" else { return }
        let url = Bundle.module.url(forResource: "long20", withExtension: "mp3", subdirectory: "media")!
        for repo in ["mlx-community/Qwen3-ASR-0.6B-4bit", "mlx-community/Qwen3-ASR-1.7B-4bit"] {
            let model = try await Qwen3ASRModel.fromPretrained(repo)
            let (_, audio) = try loadAudioArray(from: url, sampleRate: model.sampleRate)
            _ = model.generate(audio: audio[0..<(model.sampleRate * 10)], language: "English")
            for chunk: Float in [60, 300, 1200] {
                let start = Date()
                let out = model.generate(audio: audio, language: "English", chunkDuration: chunk)
                let wall = Date().timeIntervalSince(start)
                print("BENCH249 {\"model\":\"\(repo)\",\"chunk_s\":\(Int(chunk)),\"wall_s\":\(String(format: "%.1f", wall)),\"peak_gb\":\(String(format: "%.2f", out.peakMemoryUsage)),\"prompt_tokens\":\(out.promptTokens),\"gen_tokens\":\(out.generationTokens),\"prompt_tps\":\(String(format: "%.1f", out.promptTps)),\"gen_tps\":\(String(format: "%.1f", out.generationTps))}")
                print("TEXT249 \(repo) \(Int(chunk)) \(out.text.replacingOccurrences(of: "\n", with: " "))")
            }
        }
    }
}
