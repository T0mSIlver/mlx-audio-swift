import Foundation
import MLX
import MLXNN

struct VoxtralRealtimeDecoderKVCache {
    var keys: MLXArray   // [kv_len, n_kv_heads * head_dim]
    var values: MLXArray // [kv_len, n_kv_heads * head_dim]
    var positionOffset: Int
}

func voxtralComputeTimeEmbedding(
    tValue: Float,
    dim: Int,
    theta: Float = 10000.0
) -> MLXArray {
    let halfDim = dim / 2
    let invFreq = MLX.exp(
        -log(theta) * MLXArray(0..<halfDim).asType(.float32) / Float(halfDim)
    )
    let emb = tValue * invFreq
    return MLX.concatenated([MLX.cos(emb), MLX.sin(emb)], axis: 0)
}

final class VoxtralRealtimeAdaRMSNorm: Module {
    @ModuleInfo(key: "ada_down") var adaDown: Linear
    @ModuleInfo(key: "ada_up") var adaUp: Linear

    init(dim: Int, bottleneckDim: Int) {
        self._adaDown.wrappedValue = Linear(dim, bottleneckDim, bias: false)
        self._adaUp.wrappedValue = Linear(bottleneckDim, dim, bias: false)
    }

    func computeScale(tCond: MLXArray) -> MLXArray {
        let hidden = gelu(adaDown(tCond))
        return adaUp(hidden)
    }

    func callAsFunction(_ x: MLXArray, adaScale: MLXArray) -> MLXArray {
        // Cast the float32 adaScale down so it doesn't promote the fp16 hidden state.
        x * (1.0 + adaScale.asType(x.dtype))
    }
}

final class VoxtralRealtimeDecoderAttention: Module {
    let nHeads: Int
    let nKvHeads: Int
    let headDim: Int
    let slidingWindow: Int
    let ropeTheta: Float
    let scale: Float

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(_ config: VoxtralRealtimeDecoderConfig) {
        nHeads = config.nHeads
        nKvHeads = config.nKvHeads
        headDim = config.headDim
        slidingWindow = config.slidingWindow
        ropeTheta = config.ropeTheta
        scale = pow(Float(config.headDim), -0.5)

        let qDim = config.nHeads * config.headDim
        let kvDim = config.nKvHeads * config.headDim

        self._wq.wrappedValue = Linear(config.dim, qDim, bias: false)
        self._wk.wrappedValue = Linear(config.dim, kvDim, bias: false)
        self._wv.wrappedValue = Linear(config.dim, kvDim, bias: false)
        self._wo.wrappedValue = Linear(qDim, config.dim, bias: false)

    }

    /// Interleaved-RoPE cos/sin tables for `positions`.
    ///
    /// Every decoder layer rotates by the same tables, so the decoder computes
    /// them once per forward pass and hands them to each layer — rebuilding them
    /// inside all `nLayers` attentions cost hundreds of identical tiny kernel
    /// launches per decoded token. The operations are unchanged, so the tables
    /// (and therefore the layer outputs) are bit-identical to the previous
    /// per-layer construction.
    static func ropeFrequencies(
        positions: MLXArray,
        headDim: Int,
        ropeTheta: Float
    ) -> (MLXArray, MLXArray) {
        let idx = MLXArray(stride(from: 0, to: headDim, by: 2)).asType(.float32)
        let ropeInvFreq = 1.0 / MLX.pow(MLXArray(ropeTheta), idx / Float(headDim))
        let angles = positions.asType(.float32).expandedDimensions(axis: 1) * ropeInvFreq.expandedDimensions(axis: 0)
        return (MLX.cos(angles), MLX.sin(angles))
    }

    func callAsFunction(
        _ x: MLXArray,
        positions: MLXArray,
        ropeCos: MLXArray,
        ropeSin: MLXArray,
        cache: VoxtralRealtimeDecoderKVCache?
    ) -> (MLXArray, VoxtralRealtimeDecoderKVCache) {
        let seqLen = x.shape[0]

        var q = wq(x)
        var k = wk(x)
        var v = wv(x)

        q = voxtralApplyInterleavedRoPE(q, cos: ropeCos, sin: ropeSin, nHeads: nHeads, headDim: headDim)
        k = voxtralApplyInterleavedRoPE(k, cos: ropeCos, sin: ropeSin, nHeads: nKvHeads, headDim: headDim)

        var positionOffset = cache?.positionOffset ?? 0
        if let cache {
            k = MLX.concatenated([cache.keys, k], axis: 0)
            v = MLX.concatenated([cache.values, v], axis: 0)
        }

        var kvLen = k.shape[0]
        if kvLen > slidingWindow {
            let trim = kvLen - slidingWindow
            k = k[trim...]
            v = v[trim...]
            kvLen = slidingWindow
            positionOffset += trim
        }

        let newCache = VoxtralRealtimeDecoderKVCache(
            keys: k,
            values: v,
            positionOffset: positionOffset
        )

        let q4 = q.reshaped(1, seqLen, nHeads, headDim).transposed(0, 2, 1, 3)
        let k4 = k.reshaped(1, kvLen, nKvHeads, headDim).transposed(0, 2, 1, 3)
        let v4 = v.reshaped(1, kvLen, nKvHeads, headDim).transposed(0, 2, 1, 3)

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
        if seqLen == 1 {
            maskMode = .none
        } else if seqLen <= slidingWindow && cache == nil {
            maskMode = .causal
        } else {
            let qPos = positions.expandedDimensions(axis: 1)
            let kPos = MLXArray(positionOffset..<(positionOffset + kvLen)).asType(.int32).expandedDimensions(axis: 0)
            let causal = kPos .<= qPos
            let window = kPos .>= (qPos - MLXArray(Int32(slidingWindow - 1)))
            let allowed = logicalAnd(causal, window)
            // Match the activation dtype: a float32 mask over fp16 q/k aborts SDPA.
            let mask = MLX.where(allowed, MLXArray(0.0), MLXArray(-1e9)).asType(q.dtype)
            maskMode = .array(mask)
        }

        let attn = MLXFast.scaledDotProductAttention(
            queries: q4,
            keys: k4,
            values: v4,
            scale: scale,
            mask: maskMode
        )

        let out = attn.transposed(0, 2, 1, 3).reshaped(seqLen, nHeads * headDim)
        return (wo(out), newCache)
    }
}

final class VoxtralRealtimeDecoderLayer: Module {
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "attention") var attention: VoxtralRealtimeDecoderAttention
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    @ModuleInfo(key: "ada_rms_norm_t_cond") var adaRmsNormTCond: VoxtralRealtimeAdaRMSNorm?

    @ModuleInfo(key: "feed_forward_w1") var feedForwardW1: Linear
    @ModuleInfo(key: "feed_forward_w3") var feedForwardW3: Linear
    @ModuleInfo(key: "feed_forward_w2") var feedForwardW2: Linear

    init(_ config: VoxtralRealtimeDecoderConfig) {
        self._attentionNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        self._attention.wrappedValue = VoxtralRealtimeDecoderAttention(config)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)

        if config.adaRmsNormTCond {
            self._adaRmsNormTCond.wrappedValue = VoxtralRealtimeAdaRMSNorm(
                dim: config.dim,
                bottleneckDim: config.adaRmsNormTCondDim
            )
        } else {
            self._adaRmsNormTCond.wrappedValue = nil
        }

        self._feedForwardW1.wrappedValue = Linear(config.dim, config.hiddenDim, bias: false)
        self._feedForwardW3.wrappedValue = Linear(config.dim, config.hiddenDim, bias: false)
        self._feedForwardW2.wrappedValue = Linear(config.hiddenDim, config.dim, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray,
        positions: MLXArray,
        ropeCos: MLXArray,
        ropeSin: MLXArray,
        adaScale: MLXArray?,
        cache: VoxtralRealtimeDecoderKVCache?
    ) -> (MLXArray, VoxtralRealtimeDecoderKVCache) {
        var out = x

        var h = attentionNorm(out)
        let attn = attention(h, positions: positions, ropeCos: ropeCos, ropeSin: ropeSin, cache: cache)
        h = attn.0
        out = out + h

        h = ffnNorm(out)
        if let adaScale, let ada = adaRmsNormTCond {
            h = ada(h, adaScale: adaScale)
        }

        let gate = silu(feedForwardW1(h))
        let up = feedForwardW3(h)
        out = out + feedForwardW2(gate * up)

        return (out, attn.1)
    }
}

final class VoxtralRealtimeDecoder: Module {
    let config: VoxtralRealtimeDecoderConfig

    @ModuleInfo(key: "tok_embeddings") var tokEmbeddings: Embedding
    @ModuleInfo(key: "layers") var layers: [VoxtralRealtimeDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    var adaScales: [MLXArray?]?

    init(_ config: VoxtralRealtimeDecoderConfig) {
        self.config = config
        self._tokEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.dim
        )
        self._layers.wrappedValue = (0..<config.nLayers).map { _ in
            VoxtralRealtimeDecoderLayer(config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
    }

    func precomputeAdaScales(_ tCond: MLXArray) {
        var scales: [MLXArray?] = []
        scales.reserveCapacity(layers.count)

        for layer in layers {
            if let ada = layer.adaRmsNormTCond {
                scales.append(ada.computeScale(tCond: tCond))
            } else {
                scales.append(nil)
            }
        }

        adaScales = scales
    }

    /// Quantize the tied embedding/LM head in place after the fp16 checkpoint loads.
    ///
    /// The serving checkpoints store `tok_embeddings` unquantized (their converter
    /// skips embeddings), so every decoded token paid a 768 MiB fp16 gemv for the
    /// logits row — the largest single phase of a live-streaming step. Converting
    /// after load applies the same deterministic affine quantization a converted
    /// checkpoint would carry (voxmlx has always served a quantized tied head), cuts
    /// the head read to the quantized fast path, and drops the fp16 copy from
    /// residency. Logits change within quantization error — real-model eval-gated,
    /// not bit-identical.
    func quantizeTiedHeadIfNeeded(groupSize: Int, bits: Int) {
        guard !(tokEmbeddings is QuantizedEmbedding) else { return }
        guard tokEmbeddings.weight.dim(-1) % groupSize == 0 else { return }
        // Swap through the library's module-replacement machinery — ModuleInfo
        // properties trap on direct post-init mutation. The default apply converts
        // the Embedding via its Quantizable conformance (same QuantizedEmbedding).
        quantize(model: self, groupSize: groupSize, bits: bits) { path, module in
            path == "tok_embeddings" && module is Embedding
        }
    }

    func embedToken(tokenId: Int) -> MLXArray {
        // Module lookup (not a raw `weight` row): the weight is bit-packed once the
        // tied head is quantized, and the module dequantizes per-row.
        tokEmbeddings(MLXArray([Int32(tokenId)])).squeezed(axis: 0)
    }

    func embedTokens(_ tokenIds: MLXArray) -> MLXArray {
        tokEmbeddings(tokenIds)
    }

    func callAsFunction(
        _ embeds: MLXArray,
        startPos: Int,
        cache: [VoxtralRealtimeDecoderKVCache?]? = nil
    ) -> (MLXArray, [VoxtralRealtimeDecoderKVCache?]) {
        var h = embeds
        let seqLen = h.shape[0]
        let positions = MLXArray(startPos..<(startPos + seqLen)).asType(.int32)
        // Shared by every layer — see `VoxtralRealtimeDecoderAttention.ropeFrequencies`.
        let (ropeCos, ropeSin) = VoxtralRealtimeDecoderAttention.ropeFrequencies(
            positions: positions,
            headDim: config.headDim,
            ropeTheta: config.ropeTheta
        )

        var newCache: [VoxtralRealtimeDecoderKVCache?] = []
        newCache.reserveCapacity(layers.count)

        for i in layers.indices {
            let layerCache = cache?[i]
            let adaScale = adaScales?[i]
            let next = layers[i](
                h, positions: positions, ropeCos: ropeCos, ropeSin: ropeSin,
                adaScale: adaScale, cache: layerCache)
            h = next.0
            newCache.append(next.1)
        }

        h = norm(h)
        return (h, newCache)
    }

    func logits(_ h: MLXArray) -> MLXArray {
        // `asLinear` dispatches to the module's projection: a plain fp16 embedding
        // reproduces the previous matmul(h, weight.T) exactly; a quantized tied head
        // takes the quantized-matmul fast path. Callers pass a single hidden row, so
        // lift to rank 2 for the projection and drop the batch axis after.
        tokEmbeddings.asLinear(h.expandedDimensions(axis: 0)).squeezed(axis: 0)
    }
}
