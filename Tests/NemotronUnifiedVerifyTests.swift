import Foundation
import MLX
import Testing

@testable import MLXAudioCore
@testable import MLXAudioSTT

/// Verification driver, not a unit test: runs a real Nemotron Unified checkpoint over a
/// clip manifest and writes one JSON line per (clip, latency) for comparison with NeMo
/// and the Python port. Skipped unless NEMOTRON_VERIFY_MANIFEST, NEMOTRON_VERIFY_MODEL
/// and NEMOTRON_VERIFY_OUT are set.
struct NemotronUnifiedVerifyTests {
    private struct Row: Decodable {
        let id: String
        let file: String
        let lang: String
        let latencies: [Int]
        let offline: Bool?
    }

    @Test func runManifest() throws {
        let env = ProcessInfo.processInfo.environment
        guard let manifestPath = env["NEMOTRON_VERIFY_MANIFEST"],
              let modelDir = env["NEMOTRON_VERIFY_MODEL"],
              let outPath = env["NEMOTRON_VERIFY_OUT"]
        else {
            print("Skipping Nemotron Unified verification: NEMOTRON_VERIFY_* not set.")
            return
        }
        let label = env["NEMOTRON_VERIFY_LABEL"] ?? "swift"
        let only = Set((env["NEMOTRON_VERIFY_LATENCIES"] ?? "").split(separator: ",").compactMap { Int($0) })
        let kinds = Set((env["NEMOTRON_VERIFY_KINDS"] ?? "").split(separator: ",").map(String.init))
        let push = Int(env["NEMOTRON_VERIFY_PUSH"] ?? "") ?? 1280

        let loadStart = Date()
        let model = try NemotronASRModel.fromDirectory(URL(fileURLWithPath: modelDir))
        print("load \(Date().timeIntervalSince(loadStart))s")
        let manifestURL = URL(fileURLWithPath: manifestPath)
        let root = manifestURL.deletingLastPathComponent()
        let lines = try String(contentsOf: manifestURL, encoding: .utf8).split(separator: "\n")
        FileManager.default.createFile(atPath: outPath, contents: nil)
        let out = try FileHandle(forWritingTo: URL(fileURLWithPath: outPath))
        defer { try? out.close() }

        func write(_ object: [String: Any]) throws {
            var data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
            data.append(0x0A)
            out.write(data)
        }

        for line in lines {
            let row = try JSONDecoder().decode(Row.self, from: Data(line.utf8))
            if !kinds.isEmpty {
                let kind = (try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])?["kind"] as? String
                if !kinds.contains(kind ?? "") { continue }
            }
            let (_, audio) = try loadAudioArray(from: root.appendingPathComponent(row.file), sampleRate: 16000)
            let samples: [Float] = audio.asArray(Float.self)

            if row.offline ?? false {
                let start = Date()
                let output = model.generate(
                    audio: MLXArray(samples), generationParameters: STTGenerateParameters(language: row.lang)
                )
                try write([
                    "id": row.id, "latency": "offline", "impl": "swift-\(label)",
                    "text": output.text, "seconds": Date().timeIntervalSince(start),
                ])
            }
            for latency in row.latencies where only.isEmpty || only.contains(latency) {
                let start = Date()
                let session = model.makeStreamSession(language: row.lang, chunkMs: latency)
                var deltas = ""
                var i = 0
                var steps = 0
                var maxStep = 0.0
                var maxStepAt = 0.0
                // Per minute of audio: mean push time and active memory, to see drift.
                var minuteMs: [Double] = [], minuteMB: [Double] = []
                var bucketSum = 0.0, bucketCount = 0
                while i < samples.count {
                    let end = min(i + push, samples.count)
                    let stepStart = Date()
                    deltas += session.step(Array(samples[i..<end])).text
                    let took = Date().timeIntervalSince(stepStart)
                    if took > maxStep { maxStep = took; maxStepAt = Double(end) / 16000 }
                    bucketSum += took
                    bucketCount += 1
                    if end / (60 * 16000) > i / (60 * 16000) {
                        minuteMs.append(bucketSum / Double(bucketCount) * 1000)
                        minuteMB.append(Double(Memory.activeMemory) / 1_048_576)
                        bucketSum = 0
                        bucketCount = 0
                    }
                    steps += 1
                    i = end
                }
                let finishStart = Date()
                deltas += session.finish().text
                try write([
                    "id": row.id, "latency": latency, "impl": "swift-\(label)",
                    "tokens": session.tokens, "text": session.text, "delta_text": deltas,
                    "seconds": Date().timeIntervalSince(start), "pushes": steps, "max_push_s": maxStep,
                    "max_push_at_s": maxStepAt, "finish_s": Date().timeIntervalSince(finishStart),
                    "minute_mean_push_ms": minuteMs, "minute_active_mb": minuteMB,
                    "peak_mb": Double(Memory.peakMemory) / 1_048_576,
                ])
                Memory.clearCache()
            }
            print("done \(row.id)")
        }
    }
}
