import Foundation
@preconcurrency import MLX
import MLXNN
import Testing
import Tokenizers

import MLXAudioSTT

/// A random-weight Qwen3-ASR small enough to stream without a checkpoint.
private func makeTinyQwen3ASRModel() throws -> Qwen3ASRModel {
    let config = Qwen3ASRConfig(
        audioConfig: Qwen3AudioEncoderConfig(
            encoderLayers: 1, encoderAttentionHeads: 2, encoderFfnDim: 32, dModel: 16,
            outputDim: 16, downsampleHiddenSize: 8
        ),
        textConfig: Qwen3TextConfig(
            hiddenSize: 16, intermediateSize: 32, numHiddenLayers: 1,
            numAttentionHeads: 2, numKeyValueHeads: 1, headDim: 8
        )
    )
    let model = Qwen3ASRModel(config)
    eval(model.parameters())

    // Only the prompt's special tokens; other text encodes to <unk>.
    model.tokenizer = try AutoTokenizer.from(
        tokenizerConfig: ["tokenizer_class": "GPT2Tokenizer", "unk_token": "<unk>", "fuse_unk": true],
        tokenizerData: [
            "model": [
                "type": "BPE", "unk_token": "<unk>", "merges": [],
                "vocab": [
                    "<unk>": 0, "<|im_start|>": 151_644, "<|im_end|>": 151_645,
                    "<|audio_start|>": 151_669, "<|audio_end|>": 151_670, "<|audio_pad|>": 151_676,
                ],
            ],
            "added_tokens": [
                ["id": 0, "content": "<unk>", "special": true],
                ["id": 151_644, "content": "<|im_start|>", "special": true],
                ["id": 151_645, "content": "<|im_end|>", "special": true],
                ["id": 151_669, "content": "<|audio_start|>", "special": true],
                ["id": 151_670, "content": "<|audio_end|>", "special": true],
                ["id": 151_676, "content": "<|audio_pad|>", "special": true],
            ],
        ]
    )
    return model
}

@Suite("Qwen3-ASR streaming concurrency", .serialized)
struct Qwen3ASRStreamingConcurrencyTests {
    /// `feedAudio` runs the encoder on the caller's thread while decode passes
    /// run in detached tasks; both use the same model.
    @Test func feedAudioWhileDecoding() async throws {
        let model = try makeTinyQwen3ASRModel()
        let config = StreamingConfig(
            decodeIntervalSeconds: 0,
            boundaryDecodeIntervalSeconds: 0,
            maxTokensPerPass: 32,
            minAgreementPasses: 1,
            finalizeCompletedWindows: true
        )
        let chunk = (0..<1_600).map { Float(sin(Double($0) * 0.05)) * 0.1 }

        for _ in 0..<4 {
            let session = StreamingInferenceSession(model: model, config: config)
            let drain = Task {
                var ended = false
                for await event in session.events {
                    if case .ended = event { ended = true }
                }
                return ended
            }
            // 30 s of audio fed as fast as the caller can, so decode passes overlap it.
            for _ in 0..<300 {
                session.feedAudio(samples: chunk)
            }
            session.stop()
            #expect(await drain.value)
        }
    }
}
