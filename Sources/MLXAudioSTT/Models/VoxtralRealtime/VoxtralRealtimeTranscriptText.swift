import Foundation

/// The transcript of a stream session, built from each token's bytes as the token is
/// decoded, so a step's cost does not grow with the transcript.
///
/// `text` always equals decoding every appended byte in one pass with
/// `String(decoding:as: UTF8.self)`, replacement characters included: bytes are only
/// moved into `settled` at points where that decoder has finished a character, so
/// nothing appended later can change how they read.
struct VoxtralRealtimeTranscriptText {
    /// Text no later byte can change.
    private(set) var settled = ""
    /// The start of a UTF-8 sequence whose remaining bytes have not arrived yet.
    private var unsettledBytes: [UInt8] = []

    /// The whole transcript so far. While a character is incomplete it ends in U+FFFD,
    /// exactly as a one-pass decode of the same bytes would.
    ///
    /// While a character is incomplete, reading it builds a new string as long as the
    /// transcript.
    var text: String {
        unsettledBytes.isEmpty ? settled : settled + unsettledText
    }

    private var unsettledText: String {
        String(decoding: unsettledBytes, as: UTF8.self)
    }

    mutating func append(_ bytes: [UInt8]) {
        guard !bytes.isEmpty else { return }
        let pending = unsettledBytes + bytes
        let cut = Self.unsettledSuffixStart(pending)
        settled += String(decoding: pending[..<cut], as: UTF8.self)
        unsettledBytes = Array(pending[cut...])
    }

    /// Where the trailing unfinished UTF-8 sequence in `bytes` starts, or `bytes.count`
    /// when the bytes end on a finished character or on bytes that are invalid whatever
    /// follows.
    ///
    /// Cutting there is safe for a one-pass decoder: a lead byte always starts a new
    /// character, so the decoder has finished everything before it, and a sequence
    /// that already has all its continuation bytes reads the same whatever comes next.
    static func unsettledSuffixStart(_ bytes: [UInt8]) -> Int {
        var leadIndex = bytes.count
        var continuationCount = 0
        while leadIndex > 0, continuationCount < 3, bytes[leadIndex - 1] & 0xC0 == 0x80 {
            leadIndex -= 1
            continuationCount += 1
        }
        guard leadIndex > 0 else { return bytes.count }

        let expectedContinuations: Int
        switch bytes[leadIndex - 1] {
        case 0xC2...0xDF: expectedContinuations = 1
        case 0xE0...0xEF: expectedContinuations = 2
        case 0xF0...0xF4: expectedContinuations = 3
        default: return bytes.count  // ASCII, a stray continuation, or never valid
        }
        return continuationCount < expectedContinuations ? leadIndex - 1 : bytes.count
    }

    /// A position in the transcript to measure the next delta from.
    struct Mark {
        fileprivate let settledUTF8Count: Int
        fileprivate let lastSettledCharacter: String
        fileprivate let unsettledText: String
    }

    var mark: Mark {
        Mark(
            settledUTF8Count: settled.utf8.count,
            lastSettledCharacter: settled.last.map(String.init) ?? "",
            unsettledText: unsettledText
        )
    }

    /// The text added since `mark`: the new suffix when the transcript at `mark` is a
    /// prefix of the current one, and the whole transcript when it is not. That happens
    /// when a trailing U+FFFD became a real character, or when new text combined with
    /// the last character into one.
    ///
    /// Only the last settled character at `mark` and what follows it can compare
    /// differently, so only that tail is compared.
    func delta(since mark: Mark) -> String {
        // `settled` only grows, and it always ends on a scalar boundary, so the old UTF-8
        // count indexes a scalar boundary of the current string.
        let start = settled.utf8.index(settled.utf8.startIndex, offsetBy: mark.settledUTF8Count)
        let newlySettled = String(settled.unicodeScalars[start...])
        let oldTail = mark.lastSettledCharacter + mark.unsettledText
        let newTail = mark.lastSettledCharacter + newlySettled + unsettledText
        guard newTail.hasPrefix(oldTail) else { return text }
        return String(newTail.dropFirst(oldTail.count))
    }
}
