// Bench-only probe for Qwen3-TTS decode speed and output equality. Not part of the PR.
import CryptoKit
import Foundation
import MLX
import MLXAudioCore
import MLXAudioTTS

func sha(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

let args = CommandLine.arguments
guard args.count >= 5 else {
    print("usage: qwen3tts-probe <modelDir> <runs> <outDir> <label> [refWav refText]")
    exit(2)
}
let modelDir = URL(fileURLWithPath: args[1])
let runs = Int(args[2])!
let outDir = URL(fileURLWithPath: args[3])
let label = args[4]
let refWav = args.count > 6 ? args[5] : nil
let refText = args.count > 6 ? args[6] : nil
try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

let env = ProcessInfo.processInfo.environment
Qwen3TTSExperiment.rope = Int(env["Q3_ROPE"] ?? "0") ?? 0
Qwen3TTSExperiment.cpTable = env["Q3_CPTABLE"] == "1"
print("EXP rope=\(Qwen3TTSExperiment.rope) cpTable=\(Qwen3TTSExperiment.cpTable) opsPerBuffer=\(env["MLX_MAX_OPS_PER_BUFFER"] ?? "-")")
let model = try await Qwen3TTSModel.fromModelDirectory(modelDir)
let sampleRate = Double(model.sampleRate)
var refAudio: MaterializedArray? = nil
if let refWav {
    (_, refAudio) = try loadAudioArray(from: URL(fileURLWithPath: refWav), sampleRate: model.sampleRate)
}

let text = "The quick brown fox jumps over the lazy dog. Real-time speech needs every decoding step to finish in under eighty milliseconds."
var params = model.defaultGenerationParameters
var paramsTopK = params
paramsTopK.topK = 50
paramsTopK.topP = 0.9

struct Case { let name: String; let params: GenerateParameters; let seed: UInt64 }
let cases = [
    Case(name: "default", params: params, seed: 7),
    Case(name: "topk50-topp0.9", params: paramsTopK, seed: 11),
]

// Warm-up: kernels, caches.
MLXRandom.seed(1)
_ = try await model.generate(text: "Warm up.", voice: nil, refAudio: refAudio, refText: refText, language: nil, generationParameters: params)

for c in cases {
    for run in 0 ..< runs {
        // Non-streaming: full audio, used for the equality hash.
        MLXRandom.seed(c.seed)
        Memory.peakMemory = 0
        var info: AudioGenerationInfo?
        var tokens: [Int] = []
        let t0 = Date()
        var chunks: [MLXArray] = []
        var firstChunk: Double?
        for try await event in model.generateStream(
            text: text, voice: nil, refAudio: refAudio, refText: refText, language: nil,
            generationParameters: c.params, streamingInterval: 0.32
        ) {
            switch event {
            case .token(let id): tokens.append(id)
            case .info(let i): info = i
            case .audio(let a):
                if firstChunk == nil { firstChunk = Date().timeIntervalSince(t0) }
                chunks.append(a)
            default: break
            }
        }
        let streamWall = Date().timeIntervalSince(t0)
        let streamAudio = concatenated(chunks, axis: 0).asType(.float32)
        let streamSamples = streamAudio.asArray(Float.self)
        let streamData = streamSamples.withUnsafeBufferPointer { Data(buffer: $0) }
        let tokData = tokens.map { Int32($0) }.withUnsafeBufferPointer { Data(buffer: $0) }

        MLXRandom.seed(c.seed)
        let t1 = Date()
        let audio = try await model.generate(
            text: text, voice: nil, refAudio: refAudio, refText: refText, language: nil,
            generationParameters: c.params)
        let wall = Date().timeIntervalSince(t1)
        let samples = audio.asType(.float32).asArray(Float.self)
        let data = samples.withUnsafeBufferPointer { Data(buffer: $0) }
        if run == 0 {
            try data.write(to: outDir.appendingPathComponent("\(label)-\(c.name).f32"))
            try streamData.write(to: outDir.appendingPathComponent("\(label)-\(c.name)-stream.f32"))
            try tokData.write(to: outDir.appendingPathComponent("\(label)-\(c.name)-tokens.i32"))
        }
        let dur = Double(samples.count) / sampleRate
        let sdur = Double(streamSamples.count) / sampleRate
        let steps = info?.generationTokenCount ?? 0
        let genTime = info?.generateTime ?? 0
        print(String(
            format: "RESULT label=%@ case=%@ run=%d steps=%d steps_per_s=%.2f gen_s=%.3f stream_wall_s=%.3f stream_rtf=%.3f first_chunk_s=%.3f full_wall_s=%.3f full_rtf=%.3f audio_s=%.2f stream_audio_s=%.2f peak_gb=%.2f full_sha=%@ stream_sha=%@ tokens_sha=%@",
            label, c.name, run, steps, info?.tokensPerSecond ?? 0, genTime, streamWall, streamWall / max(sdur, 1e-9),
            firstChunk ?? -1, wall, wall / max(dur, 1e-9), dur, sdur, info?.peakMemoryUsage ?? 0,
            String(sha(data).prefix(16)), String(sha(streamData).prefix(16)), String(sha(tokData).prefix(16))))
        fflush(stdout)
    }
}

// Per-step profile (only when built with the bench-only timers).
if CommandLine.arguments.contains("--profile") || ProcessInfo.processInfo.environment["Q3_PROFILE"] != nil {
    for split in [false] {
        Qwen3TTSProfile.splitEval = split
        for _ in 0 ..< 2 {
            Qwen3TTSProfile.reset()
            MLXRandom.seed(7)
            _ = try await model.generate(text: text, voice: nil, refAudio: refAudio, refText: refText, language: nil, generationParameters: params)
            print(Qwen3TTSProfile.report("label=\(label) split=\(split)"))
            fflush(stdout)
        }
    }
    Qwen3TTSProfile.splitEval = false
}
