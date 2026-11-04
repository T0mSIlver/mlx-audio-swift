import Foundation
import MLX
import Testing

@testable import MLXAudioSTT

/// What a realtime server relies on from a Nemotron Unified (buffered) session, on a
/// tiny random model: text that only grows, at chunk sizes above one frame too;
/// independent sessions on one model; the latency rule for every 80 ms step.
struct NemotronUnifiedSessionContractTests {
    private var mlxRuntimeEnabled: Bool {
        ProcessInfo.processInfo.environment["MLXAUDIO_ENABLE_MLX_RUNTIME_TESTS"] == "1"
    }

    private static let config = """
    {
      "model_type": "nemotron_asr",
      "preprocessor": {"sample_rate": 16000, "features": 16, "n_fft": 64, "window_size": 0.004,
        "window_stride": 0.002, "window": "hann", "preemph": 0.97, "dither": 0.0, "normalize": "per_feature"},
      "encoder": {"feat_in": 16, "n_layers": 2, "d_model": 16, "n_heads": 2, "ff_expansion_factor": 2,
        "subsampling_factor": 4, "subsampling_conv_channels": 4, "conv_kernel_size": 3,
        "causal_downsampling": false, "conv_context_size": [1, 1], "conv_norm_type": "batch_norm",
        "self_attention_model": "rel_pos", "att_context_style": "chunked_limited_with_rc",
        "att_context_size": [[-1, -1, -1]], "pos_emb_max_len": 256, "use_bias": true, "xscaling": true},
      "prompt": {"num_prompts": 4, "prompt_hidden": 16,
        "prompt_dictionary": {"en-US": 0, "en-GB": 1, "de-DE": 2, "auto": 3}},
      "decoder": {"pred_hidden": 8, "pred_rnn_layers": 1, "vocab_size": 6, "blank_as_pad": true},
      "joint": {"joint_hidden": 8, "activation": "relu", "encoder_hidden": 16, "pred_hidden": 8, "num_classes": 6},
      "vocabulary": ["<unk>", "▁a", "▁hello", "▁world", "!", "b"],
      "default_language": "auto",
      "default_att_context_size": [6, 1, 3],
      "max_symbols": 3
    }
    """

    private func model() throws -> NemotronASRModel {
        let config = try JSONDecoder().decode(NemotronASRConfig.self, from: Data(Self.config.utf8))
        let model = NemotronASRModel(config)
        eval(model.parameters())
        model.train(false)
        return model
    }

    private func audio(_ count: Int, seed: UInt64 = 7) -> [Float] {
        guard count > 0 else { return [] }
        return MLXRandom.normal([count], key: MLXRandom.key(seed)).asArray(Float.self).map { $0 * 0.1 }
    }

    /// Feeds `feed`-sample steps; checks after every step that the text only grows by
    /// the returned delta. Returns the final text and tokens.
    private func run(
        _ session: NemotronASRStreamSession, _ samples: [Float], feed: Int
    ) -> (text: String, tokens: [Int]) {
        var joined = ""
        var i = 0
        while i < samples.count {
            let end = min(i + feed, samples.count)
            let before = session.text
            joined += session.step(Array(samples[i..<end])).text
            #expect(session.text.hasPrefix(before))
            #expect(joined == session.text)
            i = end
        }
        let before = session.text
        joined += session.finish().text
        #expect(session.text.hasPrefix(before))
        #expect(joined == session.text)
        return (session.text, session.tokens)
    }

    @Test func textOnlyGrowsAndFeedSizeDoesNotMatterAtEveryChunkSize() throws {
        guard mlxRuntimeEnabled else { return }
        let model = try model()
        let samples = audio(128 * 60 + 77)
        for latency in [160, 320, 560, 1120] {
            let one = run(model.makeStreamSession(language: "en-US", chunkMs: latency), samples, feed: 128)
            let odd = run(model.makeStreamSession(language: "en-US", chunkMs: latency), samples, feed: 61)
            let big = run(model.makeStreamSession(language: "en-US", chunkMs: latency), samples, feed: 128 * 23)
            #expect(one.tokens == odd.tokens)
            #expect(one.tokens == big.tokens)
        }
    }

    @Test func interleavedSessionsDoNotShareState() throws {
        guard mlxRuntimeEnabled else { return }
        let model = try model()
        let a = audio(128 * 50, seed: 1), b = audio(128 * 50, seed: 2)
        let aloneA = run(model.makeStreamSession(language: "en-US", chunkMs: 320), a, feed: 128)
        let aloneB = run(model.makeStreamSession(language: "de", chunkMs: 560), b, feed: 128)
        let sa = model.makeStreamSession(language: "en-US", chunkMs: 320)
        let sb = model.makeStreamSession(language: "de", chunkMs: 560)
        var i = 0
        while i < a.count {
            _ = sa.step(Array(a[i..<(i + 128)]))
            _ = sb.step(Array(b[i..<(i + 128)]))
            i += 128
        }
        _ = sa.finish()
        _ = sb.finish()
        #expect(sa.tokens == aloneA.tokens)
        #expect(sb.tokens == aloneB.tokens)
    }

    @Test func everyEightyMillisecondLatencyKeepsItsTotal() {
        for frames in 2...26 {
            let c = NemotronASRBufferedContext.forLatency(frames * 80)
            #expect(c.chunk + c.right == frames)
            #expect(c.left == 70)
            #expect(c.chunk >= 1)
        }
        #expect(NemotronASRBufferedContext.forLatency(0) == NemotronASRBufferedContext.forLatency(160))
        #expect(NemotronASRBufferedContext.forLatency(5000) == NemotronASRBufferedContext.forLatency(2080))
        #expect(NemotronASRBufferedContext.forLatency(320) == NemotronASRBufferedContext(left: 70, chunk: 1, right: 3))
        #expect(NemotronASRBufferedContext.forLatency(2080) == NemotronASRBufferedContext(left: 70, chunk: 13, right: 13))
    }

}
