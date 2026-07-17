import Foundation
import MLX
import MLXAudioCore

enum VoxtralRealtimeAudio {
    static func computeMelFilters(
        numMelBins: Int = 128,
        windowSize: Int = 400,
        sampleRate: Int = 16000
    ) -> MLXArray {
        melFilters(
            sampleRate: sampleRate,
            nFft: windowSize,
            nMels: numMelBins,
            fMin: 0,
            fMax: 8000,
            norm: "slaney",
            melScale: .slaney
        )
    }

    static func computeMelSpectrogram(
        audio: MLXArray,
        melFilters: MLXArray,
        windowSize: Int = 400,
        hopLength: Int = 160,
        globalLogMelMax: Float = 1.5
    ) -> MLXArray {
        // Periodic Hann window uses N denominator, not N-1.
        let n = MLXArray(0..<windowSize).asType(.float32)
        let window = 0.5 * (1.0 - cos((2.0 * Float.pi * n) / Float(windowSize)))

        let audio1D: MLXArray
        if audio.ndim > 1 {
            audio1D = audio.reshaped([-1])
        } else {
            audio1D = audio
        }

        let paddedAudio = reflectPadCenter(audio1D, pad: windowSize / 2)
        let nSamples = paddedAudio.shape[0]
        let nFrames = 1 + max(0, (nSamples - windowSize) / hopLength)

        if nFrames <= 0 {
            return MLXArray.zeros([melFilters.shape[1], 0], type: Float.self)
        }

        let frameIdx = asStrided(
            paddedAudio,
            [nFrames, windowSize],
            strides: [hopLength, 1],
            offset: 0
        )

        let frames = frameIdx * window.expandedDimensions(axis: 0)
        let spectrum = MLXFFT.rfft(frames, axis: -1)
        var magnitudes = MLX.abs(spectrum).square()

        // Match reference: drop last frame, then transpose to [freq, frames].
        if magnitudes.shape[0] > 0 {
            magnitudes = magnitudes[0..<(magnitudes.shape[0] - 1), 0...]
        }
        magnitudes = magnitudes.transposed(1, 0)

        var melSpec = MLX.matmul(melFilters.transposed(1, 0), magnitudes)
        melSpec = MLX.maximum(melSpec, MLXArray(Float(1e-10)))
        var logSpec = MLX.log10(melSpec)

        let minVal = globalLogMelMax - 8.0
        logSpec = MLX.maximum(logSpec, MLXArray(minVal))
        logSpec = (logSpec + MLXArray(Float(4.0))) / MLXArray(Float(4.0))
        return logSpec
    }

    /// Spectral tail of `computeMelSpectrogram` (Hann window → power spectrum → mel →
    /// log-normalize), operating on already-framed samples so the streaming
    /// `VoxtralRealtimeMelStream` can run it over incrementally framed windows. Kept
    /// operation-for-operation identical to the offline function, which frames per row
    /// independently — identical window contents therefore produce identical columns.
    /// `frames` is `[nFrames, windowSize]`; returns `[nMels, nFrames]`.
    static func melColumns(
        frames: MLXArray,
        melFilters: MLXArray,
        windowSize: Int,
        globalLogMelMax: Float
    ) -> MLXArray {
        // Periodic Hann window uses N denominator, not N-1.
        let n = MLXArray(0..<windowSize).asType(.float32)
        let window = 0.5 * (1.0 - cos((2.0 * Float.pi * n) / Float(windowSize)))

        let windowed = frames * window.expandedDimensions(axis: 0)
        let spectrum = MLXFFT.rfft(windowed, axis: -1)
        let magnitudes = MLX.abs(spectrum).square().transposed(1, 0)

        var melSpec = MLX.matmul(melFilters.transposed(1, 0), magnitudes)
        melSpec = MLX.maximum(melSpec, MLXArray(Float(1e-10)))
        var logSpec = MLX.log10(melSpec)

        let minVal = globalLogMelMax - 8.0
        logSpec = MLX.maximum(logSpec, MLXArray(minVal))
        logSpec = (logSpec + MLXArray(Float(4.0))) / MLXArray(Float(4.0))
        return logSpec
    }

    private static func reflectPadCenter(_ audio: MLXArray, pad: Int) -> MLXArray {
        guard pad > 0 else { return audio.asType(.float32) }

        let samples = audio.asType(.float32).asArray(Float.self)
        guard !samples.isEmpty else {
            return MLXArray.zeros([2 * pad], type: Float.self)
        }

        func reflectIndex(_ idx: Int, count: Int) -> Int {
            if count <= 1 { return 0 }
            var i = idx
            while i < 0 || i >= count {
                if i < 0 {
                    i = -i
                } else {
                    i = 2 * count - i - 2
                }
            }
            return i
        }

        var out = [Float](repeating: 0, count: samples.count + 2 * pad)
        for i in 0..<out.count {
            let src = i - pad
            out[i] = samples[reflectIndex(src, count: samples.count)]
        }

        return MLXArray(out)
    }
}

/// Incremental counterpart of `VoxtralRealtimeAudio.computeMelSpectrogram` for the
/// streaming session: O(chunk) work per call instead of reframing the whole stream.
///
/// The offline path reflect-pads the (already zero-padded) sample stream by
/// `windowSize / 2` on both sides, frames it at `hopLength`, and drops the last frame —
/// so frame `t` covers stream samples `[t*hop - window/2, t*hop - window/2 + window)`.
/// This type reproduces those frames bit-for-bit by carrying the not-yet-framed suffix
/// of the stream between calls:
///   * The stream always begins with the session's `nLeftPadTokens` of zero samples
///     (`>= window/2`), so the offline leading reflect pad is `window/2` zeros — the
///     carry is seeded with exactly those.
///   * A frame is emitted only once its full window is buffered, so every emitted
///     frame is final: later audio can never land inside an already-emitted window.
///   * The stream always ends with `>= window` zero samples (the offline right-pad),
///     so the offline trailing reflect pad is zeros too; of it, only the
///     `window - hop - window/2` samples past the stream end are ever read by kept
///     frames (the offline path drops the final frame). `finishTailPadCount` is that
///     count — the session appends that many zeros after the right-pad on `finish()`,
///     after which the emitted frame count equals the offline count exactly.
struct VoxtralRealtimeMelStream {
    private let melFilters: MLXArray
    private let windowSize: Int
    private let hopLength: Int
    private let globalLogMelMax: Float
    private var carry: [Float]

    /// Mel frames emitted so far (absolute frame index of the next frame).
    private(set) var framesEmitted = 0

    init(
        leftPadSamples: Int,
        melFilters: MLXArray,
        windowSize: Int = 400,
        hopLength: Int = 160,
        globalLogMelMax: Float = 1.5
    ) {
        // The offline leading reflect pad mirrors stream indices 1...windowSize/2,
        // so zero-seeding is exact only when the left pad strictly exceeds
        // windowSize/2 (at exactly windowSize/2 the mirror reads the first real
        // sample).
        precondition(
            leftPadSamples > windowSize / 2,
            "left pad must cover the reflect pad for the zero-seeded carry to be exact"
        )
        self.melFilters = melFilters
        self.windowSize = windowSize
        self.hopLength = hopLength
        self.globalLogMelMax = globalLogMelMax
        // Leading reflect pad (window/2 zeros, since the stream starts with zeros)
        // followed by the stream's left-pad zeros.
        self.carry = [Float](repeating: 0, count: windowSize / 2 + leftPadSamples)
    }

    /// Zero samples the offline path reads past the end of the stream (from the
    /// trailing reflect pad of the zero right-pad). Append on `finish()` only.
    var finishTailPadCount: Int { windowSize - hopLength - windowSize / 2 }

    /// Ingest new stream samples and return the mel columns (`[nMels, nNewFrames]`)
    /// for every newly completed window; empty while less than one window is pending.
    mutating func append(_ samples: [Float]) -> MLXArray {
        carry.append(contentsOf: samples)
        guard carry.count >= windowSize else {
            return MLXArray.zeros([melFilters.shape[1], 0], type: Float.self)
        }
        let nFrames = 1 + (carry.count - windowSize) / hopLength
        let framed = MLXArray(Array(carry[0..<((nFrames - 1) * hopLength + windowSize)]))
        carry.removeFirst(nFrames * hopLength)
        framesEmitted += nFrames

        let frames = asStrided(framed, [nFrames, windowSize], strides: [hopLength, 1], offset: 0)
        return VoxtralRealtimeAudio.melColumns(
            frames: frames,
            melFilters: melFilters,
            windowSize: windowSize,
            globalLogMelMax: globalLogMelMax
        )
    }
}
