import Foundation
import MLX
import MLXAudioCore

enum NemotronASRAudio {
    static func logMelSpectrogram(
        _ audio: MLXArray,
        config: NemotronASRPreprocessConfig
    ) -> MLXArray {
        let originalDType = audio.dtype
        var x = audio

        if config.padTo > 0 && x.shape[0] < config.padTo {
            let padLength = config.padTo - x.shape[0]
            let paddedTail = MLXArray(Array(repeating: config.padValue, count: padLength))
            x = MLX.concatenated([x, paddedTail], axis: 0)
        }

        if config.preemph > 0 && x.shape[0] > 1 {
            let first = x[0..<1]
            let rest = x[1...] - Float(config.preemph) * x[..<(x.shape[0] - 1)]
            x = MLX.concatenated([first, rest], axis: 0)
        }

        let window = makeWindow(name: config.window, winLength: config.winLength, fftLength: config.nFft)
        let stftOutput = stft(
            audio: x,
            window: window,
            nFft: config.nFft,
            hopLength: config.hopLength,
            padMode: .constant
        )

        let power = MLX.abs(stftOutput).square().asType(originalDType)
        let filters = melFilters(
            sampleRate: config.sampleRate,
            nFft: config.nFft,
            nMels: config.features,
            norm: "slaney",
            melScale: .slaney
        )

        var mel = MLX.matmul(power, filters.asType(power.dtype))
        mel = MLX.log(mel + MLXArray(config.logZeroGuardValue, dtype: mel.dtype))

        switch config.normalize.lowercased() {
        case "na", "none":
            return mel.expandedDimensions(axis: 0).asType(originalDType)
        case "per_feature":
            let mean = MLX.mean(mel, axis: 0, keepDims: true)
            let denominator = max(mel.dim(0) - 1, 1)
            let variance = MLX.sum((mel - mean).square(), axis: 0, keepDims: true) / Float(denominator)
            let std = MLX.sqrt(variance)
            mel = (mel - mean) / (std + MLXArray(1e-5, dtype: mel.dtype))
        default:
            let mean = MLX.mean(mel)
            let std = MLX.std(mel)
            mel = (mel - mean) / (std + MLXArray(1e-5, dtype: mel.dtype))
        }

        return mel.expandedDimensions(axis: 0).asType(originalDType)
    }

    /// Mel frames `[first, end)` of `logMelSpectrogram(samples)` for NA normalization,
    /// computed from only the samples those frames cover. Returns `(1, end - first, F)`.
    ///
    /// Each value equals the full computation's bit for bit, provided `first` is even
    /// and `end` is even or equals the full frame count: MLX's float RFFT packs rows
    /// (2i, 2i+1) into one complex FFT (an odd last row is packed with itself), so a
    /// row's rounding depends on its partner. Those bounds keep every partner the same.
    /// Callers must use the full path when `padTo` would pad the audio.
    ///
    /// `samples` may be a suffix of the audio: `samples[0]` is audio sample `offset`,
    /// and it must include sample `first*hop - nFft/2 - 1` (pre-emphasis context).
    static func logMelFrames(
        _ samples: [Float],
        offset: Int = 0,
        first: Int,
        end: Int,
        config: NemotronASRPreprocessConfig
    ) -> MLXArray {
        let total = offset + samples.count
        let hop = config.hopLength
        let nFft = config.nFft
        let half = nFft / 2
        let count = end - first
        // Padded-signal coordinates [first*hop, (end-1)*hop + nFft) map to samples
        // shifted by -half; outside [0, total) the STFT pad is zero.
        let lo = first * hop - half
        let hi = (end - 1) * hop - half + nFft
        let srcLo = max(lo, 0)
        let srcHi = min(hi, total)

        var parts: [MLXArray] = []
        if srcLo - lo > 0 { parts.append(MLXArray.zeros([srcLo - lo])) }
        if srcHi > srcLo {
            if config.preemph > 0 && total > 1 {
                // y[i] = x[i] - p*x[i-1], y[0] = x[0]: same elementwise ops as the full path.
                let ctxLo = max(srcLo - 1, 0)
                precondition(ctxLo >= offset, "logMelFrames: samples start after the needed context")
                let x = MLXArray(Array(samples[(ctxLo - offset)..<(srcHi - offset)]))
                let rest = x[1...] - Float(config.preemph) * x[..<(x.shape[0] - 1)]
                parts.append(srcLo == 0 ? MLX.concatenated([x[0..<1], rest], axis: 0) : rest)
            } else {
                precondition(srcLo >= offset, "logMelFrames: samples start after the needed context")
                parts.append(MLXArray(Array(samples[(srcLo - offset)..<(srcHi - offset)])))
            }
        }
        if hi - max(srcHi, lo) > 0 { parts.append(MLXArray.zeros([hi - max(srcHi, lo)])) }
        let segment = parts.count == 1 ? parts[0] : MLX.concatenated(parts, axis: 0)

        let window = makeWindow(name: config.window, winLength: config.winLength, fftLength: nFft)
        let frames = asStrided(segment, [count, nFft], strides: [hop, 1], offset: 0)
        let stftOutput = MLXFFT.rfft(frames * window, axis: 1)

        var power = MLX.abs(stftOutput).square().asType(.float32)
        // A 1-row matmul dispatches to GEMV; pad to 2 rows to stay on the GEMM kernel.
        if count == 1 { power = MLX.concatenated([power, MLXArray.zeros(like: power)], axis: 0) }
        let filters = melFilters(
            sampleRate: config.sampleRate,
            nFft: nFft,
            nMels: config.features,
            norm: "slaney",
            melScale: .slaney
        )
        var mel = MLX.matmul(power, filters.asType(power.dtype))
        if count == 1 { mel = mel[0..<1] }
        mel = MLX.log(mel + MLXArray(config.logZeroGuardValue, dtype: mel.dtype))
        return mel.expandedDimensions(axis: 0)
    }

    private static func makeWindow(name: String, winLength: Int, fftLength: Int) -> MLXArray {
        let base: MLXArray
        switch name.lowercased() {
        case "hann", "hanning":
            base = hanningWindow(size: winLength)
        case "hamming":
            base = hammingWindow(size: winLength)
        case "blackman":
            base = blackmanWindow(size: winLength)
        case "bartlett":
            base = bartlettWindow(size: winLength)
        default:
            base = hanningWindow(size: winLength)
        }

        if winLength >= fftLength {
            return base[0..<fftLength]
        }

        let left = (fftLength - winLength) / 2
        let right = fftLength - winLength - left
        return MLX.concatenated([
            MLXArray.zeros([left]),
            base,
            MLXArray.zeros([right])
        ], axis: 0)
    }

    private static func hammingWindow(size: Int) -> MLXArray {
        if size <= 1 {
            return MLXArray(Array(repeating: Float(1), count: max(size, 1)))
        }
        let denom = Float(size - 1)
        let values = (0..<size).map { n in
            Float(0.54) - Float(0.46) * cos(2 * Float.pi * Float(n) / denom)
        }
        return MLXArray(values)
    }

    private static func blackmanWindow(size: Int) -> MLXArray {
        if size <= 1 {
            return MLXArray(Array(repeating: Float(1), count: max(size, 1)))
        }
        let denom = Float(size - 1)
        let values = (0..<size).map { n in
            let k = 2 * Float.pi * Float(n) / denom
            return Float(0.42) - Float(0.5) * cos(k) + Float(0.08) * cos(2 * k)
        }
        return MLXArray(values)
    }

    private static func bartlettWindow(size: Int) -> MLXArray {
        if size <= 1 {
            return MLXArray(Array(repeating: Float(1), count: max(size, 1)))
        }
        let mid = Float(size - 1) / 2
        let values = (0..<size).map { n in
            Float(1) - abs((Float(n) - mid) / mid)
        }
        return MLXArray(values)
    }
}
