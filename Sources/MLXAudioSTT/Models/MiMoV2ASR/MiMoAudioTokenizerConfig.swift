import Foundation
import MLXAudioCore

public struct MiMoAudioTokenizerConfig: Decodable, Sendable {
    public let maxAudioSeconds: Int
    public let strideSize: Int
    public let avgPooler: Int
    public let dModel: Int
    public let scaleEmbedding: Bool
    public let kernelSize: Int
    public let activationFunction: String
    public let encoderLayers: Int
    public let encoderSkipLayerID: Int?
    public let encoderAttentionHeads: Int
    public let encoderFFNDim: Int
    public let encoderCausal: Bool
    public let encoderAttentionWindowSize: [Int]
    public let decoderLayers: Int
    public let decoderAttentionHeads: Int
    public let decoderFFNDim: Int
    public let decoderKernelSize: Int
    public let decoderStrideSize: Int
    public let decoderCausal: Bool
    public let decoderAttentionWindowSize: [Int]
    public let nfft: Int
    public let vocoderDim: Int
    public let vocoderIntermediateDim: Int
    public let vocoderNumLayers: Int
    public let nMels: Int
    public let samplingRate: Int
    public let hopLength: Int
    public let windowSize: Int
    public let vocoderPadding: String
    public let fmin: Int
    public let fmax: Int?
    public let numQuantizers: Int
    public let codebookSize: [Int]
    public let thresholdEMADeadCode: Int
    public let positionEmbeddingType: String
    public let ropeTheta: Int
    public let ropeType: String
    public let layerNormType: String
    public let vocoderAttentionHeads: Int
    public let vocoderAttentionWindowSize: [Int]
    public let quantization: BaseConfiguration.Quantization?

    enum CodingKeys: String, CodingKey {
        case maxAudioSeconds = "max_audio_seconds"
        case strideSize = "stride_size"
        case avgPooler = "avg_pooler"
        case dModel = "d_model"
        case scaleEmbedding = "scale_embedding"
        case kernelSize = "kernel_size"
        case activationFunction = "activation_function"
        case encoderLayers = "encoder_layers"
        case encoderSkipLayerID = "encoder_skip_layer_id"
        case encoderAttentionHeads = "encoder_attention_heads"
        case encoderFFNDim = "encoder_ffn_dim"
        case encoderCausal = "encoder_causal"
        case encoderAttentionWindowSize = "encoder_attn_window_size"
        case decoderLayers = "decoder_layers"
        case decoderAttentionHeads = "decoder_attention_heads"
        case decoderFFNDim = "decoder_ffn_dim"
        case decoderKernelSize = "decoder_kernel_size"
        case decoderStrideSize = "decoder_stride_size"
        case decoderCausal = "decoder_causal"
        case decoderAttentionWindowSize = "decoder_attn_window_size"
        case nfft
        case vocoderDim = "vocoder_dim"
        case vocoderIntermediateDim = "vocoder_intermediate_dim"
        case vocoderNumLayers = "vocoder_num_layers"
        case nMels = "n_mels"
        case samplingRate = "sampling_rate"
        case hopLength = "hop_length"
        case windowSize = "window_size"
        case vocoderPadding = "vocoder_padding"
        case fmin
        case fmax
        case numQuantizers = "num_quantizers"
        case codebookSize = "codebook_size"
        case thresholdEMADeadCode = "threshold_ema_dead_code"
        case positionEmbeddingType = "position_embedding_type"
        case ropeTheta = "rope_theta"
        case ropeType = "rope_type"
        case layerNormType = "ln_type"
        case vocoderAttentionHeads = "vocoder_attention_heads"
        case vocoderAttentionWindowSize = "vocoder_attn_window_size"
        case quantization
        case quantizationConfig = "quantization_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // Missing keys take the defaults of XiaomiMiMo's MiMoAudioTokenizerConfig
        // (src/mimo_audio_tokenizer/configuration_audio_tokenizer.py), as the
        // reference loader does: the MLX tokenizer repo ships only the encoder/RVQ half.

        maxAudioSeconds = try container.decodeIfPresent(Int.self, forKey: .maxAudioSeconds) ?? 1800
        strideSize = try container.decodeIfPresent(Int.self, forKey: .strideSize) ?? 2
        avgPooler = try container.decodeIfPresent(Int.self, forKey: .avgPooler) ?? 1
        dModel = try container.decodeIfPresent(Int.self, forKey: .dModel) ?? 768
        scaleEmbedding = try container.decodeIfPresent(Bool.self, forKey: .scaleEmbedding) ?? true
        kernelSize = try container.decodeIfPresent(Int.self, forKey: .kernelSize) ?? 3
        activationFunction = try container.decodeIfPresent(String.self, forKey: .activationFunction) ?? "gelu"
        encoderLayers = try container.decodeIfPresent(Int.self, forKey: .encoderLayers) ?? 8
        encoderSkipLayerID = try container.decodeIfPresent(Int.self, forKey: .encoderSkipLayerID)
        encoderAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .encoderAttentionHeads) ?? 12
        encoderFFNDim = try container.decodeIfPresent(Int.self, forKey: .encoderFFNDim) ?? 3072
        encoderCausal = try container.decodeIfPresent(Bool.self, forKey: .encoderCausal) ?? false
        encoderAttentionWindowSize = try container.decodeIfPresent([Int].self, forKey: .encoderAttentionWindowSize) ?? [-1, -1]
        decoderLayers = try container.decodeIfPresent(Int.self, forKey: .decoderLayers) ?? 8
        decoderAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .decoderAttentionHeads) ?? 12
        decoderFFNDim = try container.decodeIfPresent(Int.self, forKey: .decoderFFNDim) ?? 3072
        decoderKernelSize = try container.decodeIfPresent(Int.self, forKey: .decoderKernelSize) ?? 3
        decoderStrideSize = try container.decodeIfPresent(Int.self, forKey: .decoderStrideSize) ?? 2
        decoderCausal = try container.decodeIfPresent(Bool.self, forKey: .decoderCausal) ?? true
        decoderAttentionWindowSize = try container.decodeIfPresent([Int].self, forKey: .decoderAttentionWindowSize) ?? [-1, -1]
        nfft = try container.decodeIfPresent(Int.self, forKey: .nfft) ?? 1024
        vocoderDim = try container.decodeIfPresent(Int.self, forKey: .vocoderDim) ?? 512
        vocoderIntermediateDim = try container.decodeIfPresent(Int.self, forKey: .vocoderIntermediateDim) ?? 4096
        vocoderNumLayers = try container.decodeIfPresent(Int.self, forKey: .vocoderNumLayers) ?? 30
        nMels = try container.decodeIfPresent(Int.self, forKey: .nMels) ?? 80
        samplingRate = try container.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 24000
        hopLength = try container.decodeIfPresent(Int.self, forKey: .hopLength) ?? 240
        windowSize = try container.decodeIfPresent(Int.self, forKey: .windowSize) ?? 1024
        vocoderPadding = try container.decodeIfPresent(String.self, forKey: .vocoderPadding) ?? "same"
        fmin = try container.decodeIfPresent(Int.self, forKey: .fmin) ?? 0
        fmax = try container.decodeIfPresent(Int.self, forKey: .fmax)
        numQuantizers = try container.decodeIfPresent(Int.self, forKey: .numQuantizers) ?? 12
        codebookSize = try container.decodeIfPresent([Int].self, forKey: .codebookSize) ?? [1024]
        thresholdEMADeadCode = try container.decodeIfPresent(Int.self, forKey: .thresholdEMADeadCode) ?? 10
        positionEmbeddingType = try container.decodeIfPresent(String.self, forKey: .positionEmbeddingType) ?? "rope"
        ropeTheta = try container.decodeIfPresent(Int.self, forKey: .ropeTheta) ?? 10000
        ropeType = try container.decodeIfPresent(String.self, forKey: .ropeType) ?? "default"
        layerNormType = try container.decodeIfPresent(String.self, forKey: .layerNormType) ?? "LayerNorm"
        vocoderAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .vocoderAttentionHeads) ?? 4
        vocoderAttentionWindowSize = try container.decodeIfPresent([Int].self, forKey: .vocoderAttentionWindowSize) ?? [40, 10]

        let globalQuant = try container.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantization)
        let altGlobalQuant = try container.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantizationConfig)
        quantization = globalQuant ?? altGlobalQuant
    }

    public func activeCodebookSizes(prefixCount: Int? = nil) -> [Int] {
        guard let prefixCount else { return codebookSize }
        return Array(codebookSize.prefix(max(0, prefixCount)))
    }
}
