import Foundation
import MLX
import MLXNN

struct VoxtralRealtimeDecoderKVCacheAppendPlan: Equatable {
    static let capacityBlock = 256

    let compactionRange: Range<Int>?
    let appendRange: Range<Int>
    let capacity: Int
    let count: Int
    let positionOffset: Int
    let windowRange: Range<Int>
    let windowPositionOffset: Int
    let requiresGrowth: Bool

    static func make(
        count: Int,
        capacity: Int,
        positionOffset: Int,
        appendCount: Int,
        slidingWindow: Int
    ) -> Self {
        precondition(count >= 0 && count <= capacity)
        precondition(appendCount >= 0)
        precondition(slidingWindow > 0)

        let projectedCount = count + appendCount
        let excessAfterAppend = max(0, projectedCount - slidingWindow)
        // Let one block of invisible rows accumulate. If the storage fills first,
        // reclaim that prefix instead of growing it again.
        let shouldCompact = count > slidingWindow
            && (projectedCount > capacity || excessAfterAppend > capacityBlock)
        let compactionRange: Range<Int>? = shouldCompact
            ? ((count - slidingWindow)..<count)
            : nil
        let droppedCount = compactionRange?.lowerBound ?? 0
        let compactedCount = compactionRange?.count ?? count
        let appendRange = compactedCount..<(compactedCount + appendCount)
        let requiredCapacity = appendRange.upperBound
        let requiresGrowth = requiredCapacity > capacity
        let capacityAfterGrowth: Int
        if requiresGrowth {
            capacityAfterGrowth = max(
                capacityBlock,
                ((requiredCapacity + capacityBlock - 1) / capacityBlock) * capacityBlock
            )
        } else {
            capacityAfterGrowth = capacity
        }

        let newCount = appendRange.upperBound
        let windowStart = max(0, newCount - slidingWindow)
        let newPositionOffset = positionOffset + droppedCount
        return Self(
            compactionRange: compactionRange,
            appendRange: appendRange,
            capacity: capacityAfterGrowth,
            count: newCount,
            positionOffset: newPositionOffset,
            windowRange: windowStart..<newCount,
            windowPositionOffset: newPositionOffset + windowStart,
            requiresGrowth: requiresGrowth
        )
    }
}

/// Order-preserving decoder cache with block-grown backing storage.
/// `positionOffset` is the absolute position of storage row zero; the attention
/// offset also includes the invisible prefix at the start of `windowRange`.
final class VoxtralRealtimeDecoderKVCache {
    private(set) var keys: MLXArray   // [capacity, n_kv_heads * head_dim]
    private(set) var values: MLXArray // [capacity, n_kv_heads * head_dim]
    private(set) var count = 0
    private(set) var positionOffset = 0 // absolute position of storage row zero

    private let slidingWindow: Int

    init(keys: MLXArray, values: MLXArray, slidingWindow: Int) {
        precondition(keys.ndim == 2 && values.ndim == 2)
        precondition(keys.shape == values.shape)

        self.slidingWindow = slidingWindow
        let initialCapacity = max(
            VoxtralRealtimeDecoderKVCacheAppendPlan.capacityBlock,
            ((keys.shape[0] + VoxtralRealtimeDecoderKVCacheAppendPlan.capacityBlock - 1)
                / VoxtralRealtimeDecoderKVCacheAppendPlan.capacityBlock)
                * VoxtralRealtimeDecoderKVCacheAppendPlan.capacityBlock
        )
        self.keys = MLXArray.zeros([initialCapacity, keys.shape[1]], dtype: keys.dtype)
        self.values = MLXArray.zeros([initialCapacity, values.shape[1]], dtype: values.dtype)
        _ = append(keys: keys, values: values)
    }

    func append(keys newKeys: MLXArray, values newValues: MLXArray) -> (
        keys: MLXArray, values: MLXArray, positionOffset: Int
    ) {
        precondition(newKeys.ndim == 2 && newValues.ndim == 2)
        precondition(newKeys.shape == newValues.shape)
        precondition(newKeys.shape[1] == keys.shape[1])
        precondition(newKeys.dtype == keys.dtype && newValues.dtype == values.dtype)

        let plan = VoxtralRealtimeDecoderKVCacheAppendPlan.make(
            count: count,
            capacity: keys.shape[0],
            positionOffset: positionOffset,
            appendCount: newKeys.shape[0],
            slidingWindow: slidingWindow
        )

        if let retained = plan.compactionRange {
            keys[0..<retained.count] = keys[retained]
            values[0..<retained.count] = values[retained]
            count = retained.count
            positionOffset = plan.positionOffset
        }

        if plan.requiresGrowth {
            var grownKeys = MLXArray.zeros([plan.capacity, keys.shape[1]], dtype: keys.dtype)
            var grownValues = MLXArray.zeros([plan.capacity, values.shape[1]], dtype: values.dtype)
            if count > 0 {
                grownKeys[0..<count] = keys[0..<count]
                grownValues[0..<count] = values[0..<count]
            }
            keys = grownKeys
            values = grownValues
        }

        keys[plan.appendRange] = newKeys
        values[plan.appendRange] = newValues
        count = plan.count
        positionOffset = plan.positionOffset

        return (
            keys[plan.windowRange],
            values[plan.windowRange],
            plan.windowPositionOffset
        )
    }
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

    func callAsFunction(
        _ x: MLXArray,
        positions: MLXArray,
        startPos: Int,
        cache: VoxtralRealtimeDecoderKVCache?
    ) -> (MLXArray, VoxtralRealtimeDecoderKVCache) {
        let seqLen = x.shape[0]

        // Keep the batch axis: mlx-swift's Metal RoPE kernel misinterprets heads
        // as batches for 3-D [H, L, D] inputs when L == 1.
        let q4 = MLXFast.RoPE(
            wq(x).reshaped(1, seqLen, nHeads, headDim).transposed(0, 2, 1, 3),
            dimensions: headDim,
            traditional: true,
            base: ropeTheta,
            scale: 1.0,
            offset: startPos
        )
        let newK4 = MLXFast.RoPE(
            wk(x).reshaped(1, seqLen, nKvHeads, headDim).transposed(0, 2, 1, 3),
            dimensions: headDim,
            traditional: true,
            base: ropeTheta,
            scale: 1.0,
            offset: startPos
        )
        var k = newK4.transposed(0, 2, 1, 3).reshaped(seqLen, nKvHeads * headDim)
        var v = wv(x)

        let newCache: VoxtralRealtimeDecoderKVCache
        let window: (keys: MLXArray, values: MLXArray, positionOffset: Int)
        if let cache {
            newCache = cache
            window = cache.append(keys: k, values: v)
        } else {
            newCache = VoxtralRealtimeDecoderKVCache(
                keys: k,
                values: v,
                slidingWindow: slidingWindow
            )
            let windowRange = max(0, newCache.count - slidingWindow)..<newCache.count
            window = (
                newCache.keys[windowRange],
                newCache.values[windowRange],
                newCache.positionOffset + windowRange.lowerBound
            )
        }
        k = window.keys
        v = window.values
        let positionOffset = window.positionOffset
        let kvLen = k.shape[0]

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
            let mask = MLX.where(allowed, MLXArray(0.0), MLXArray(-1e9)).asType(q4.dtype)
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
        startPos: Int,
        adaScale: MLXArray?,
        cache: VoxtralRealtimeDecoderKVCache?
    ) -> (MLXArray, VoxtralRealtimeDecoderKVCache) {
        var out = x

        var h = attentionNorm(out)
        let attn = attention(h, positions: positions, startPos: startPos, cache: cache)
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

    func embedToken(tokenId: Int) -> MLXArray {
        tokEmbeddings.weight[tokenId]
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

        var newCache: [VoxtralRealtimeDecoderKVCache?] = []
        newCache.reserveCapacity(layers.count)

        for i in layers.indices {
            let layerCache = cache?[i]
            let adaScale = adaScales?[i]
            let next = layers[i](
                h, positions: positions, startPos: startPos,
                adaScale: adaScale, cache: layerCache)
            h = next.0
            newCache.append(next.1)
        }

        h = norm(h)
        return (h, newCache)
    }

    func logits(_ h: MLXArray) -> MLXArray {
        // Every caller passes a single hidden-state row, so the tied-head projection is
        // W·h over the embedding's stored row-major layout — mathematically identical to
        // h·Wᵀ but without rebuilding a lazy transpose of the 768 MiB fp16 weight in
        // every decoded token's graph. In live streaming (an eval boundary per step,
        // allocator under pressure) that per-token transpose node cost ~25 ms/token and
        // a ~1.5 GB transient; the batch decode loop measured the same op at ~4.8 ms.
        precondition(h.ndim == 1, "logits expects a single hidden-state row")
        return MLX.matmul(tokEmbeddings.weight, h)
    }
}
