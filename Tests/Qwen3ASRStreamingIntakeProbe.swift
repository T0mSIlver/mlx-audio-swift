// CI-only probe for T0mSIlver/mlx-audio-swift#28; not for upstream.
import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioSTT
import Testing

@Suite("Qwen3-ASR streaming intake probe", .serialized)
struct Qwen3ASRStreamingIntakeProbe {
    @Test func feedAudioLatencyAtRealTimeCadence() async throws {
        let audioURL = Bundle.module.url(forResource: "conversational_a", withExtension: "wav", subdirectory: "media")!
        let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16_000)
        let clip = audio.asArray(Float.self)
        let samples = clip + clip + clip
        let model = try await Qwen3ASRModel.fromPretrained("mlx-community/Qwen3-ASR-0.6B-4bit")

        let chunk = 1_600  // 100 ms, the localvoxtral step cadence
        let session = StreamingInferenceSession(model: model, config: StreamingConfig())
        let drain = Task { () -> (String, Int) in
            var text = ""
            var updates = 0
            for await event in session.events {
                switch event {
                case .displayUpdate: updates += 1
                case .ended(let full): text = full
                default: break
                }
            }
            return (text, updates)
        }

        var feedMs: [Double] = []
        let clock = ContinuousClock()
        var deadline = clock.now
        var offset = 0
        while offset < samples.count {
            let end = min(offset + chunk, samples.count)
            let t0 = clock.now
            session.feedAudio(samples: Array(samples[offset..<end]))
            let d = clock.now - t0
            feedMs.append(Double(d.components.seconds) * 1e3 + Double(d.components.attoseconds) / 1e15)
            offset = end
            deadline += .milliseconds(100)
            try await Task.sleep(until: deadline, clock: clock)
        }
        let stopStart = clock.now
        session.stop()
        let (text, updates) = await drain.value
        let stopDuration = clock.now - stopStart

        let sorted = feedMs.sorted()
        func pct(_ p: Double) -> Double { sorted[min(sorted.count - 1, Int(Double(sorted.count) * p))] }
        let over100 = feedMs.filter { $0 > 100 }
        print("PROBE feeds=\(feedMs.count) audioSec=\(Double(samples.count) / 16_000)")
        print(String(format: "PROBE feedMs p50=%.2f p95=%.2f p99=%.2f max=%.2f sum=%.1f", pct(0.5), pct(0.95), pct(0.99), sorted.last!, feedMs.reduce(0, +)))
        print("PROBE feeds>100ms=\(over100.count) worst=\(over100.sorted(by: >).prefix(10).map { String(format: "%.0f", $0) })")
        print("PROBE lateness: wallclock behind real time at end = \(clock.now - deadline)")
        print("PROBE stop=\(stopDuration) displayUpdates=\(updates)")
        print("PROBE text=\(text)")
        #expect(!text.isEmpty)
    }
}
