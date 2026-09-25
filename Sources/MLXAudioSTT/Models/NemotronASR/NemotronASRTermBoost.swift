import Foundation

// Decode-time term boosting for the greedy RNN-T.
//
// A caller hands the session a short list of terms it expects to hear (project
// names, identifiers, jargon). While decoding, a word piece that starts or
// continues one of those terms gets a bonus added to its joint logit before the
// argmax. Nothing is retrained and the encoder is untouched.
//
// Matching is on the decoded text, not on token ids: a piece counts when the
// lowercased text it appends extends a lowercased term, so every tokenization of
// a term is boosted, whatever casing the model prefers.
//
// Three rules keep the boost from inventing words:
//   * Blank is never boosted, and nothing is boosted when the unbiased argmax is
//     blank. The boost only chooses *which* piece is emitted, never *whether* one
//     is, so it cannot write text ahead of the audio.
//   * A piece competes only when its logit is within `margin` of the unbiased
//     top logit.
//   * The first piece of a term gets a smaller bonus than a piece continuing a
//     match already under way. Output is append-only, so a wrong first piece is
//     the costly mistake.

/// Bonus sizes for `NemotronASRStreamSession.setBoostTerms(_:config:)`, in logits.
public struct NemotronASRTermBoostConfig: Sendable, Equatable {
    /// Added to a piece that starts a listed term at a word boundary.
    public var firstTokenBoost: Float
    /// Added to a piece that continues a term whose earlier pieces were emitted.
    public var continuationBoost: Float
    /// Only pieces whose logit is within this distance of the unbiased top
    /// logit can take its place.
    public var margin: Float

    public init(firstTokenBoost: Float = 1.5, continuationBoost: Float = 3.0, margin: Float = 4.0) {
        self.firstTokenBoost = firstTokenBoost
        self.continuationBoost = continuationBoost
        self.margin = margin
    }
}

final class NemotronASRTermBooster {
    let config: NemotronASRTermBoostConfig
    /// Lowercased text each token appends, with "▁" as a space; nil for tokens
    /// that never take part in a match (blank, language tags, other specials).
    private let pieceText: [String?]
    /// Every prefix of every term key. A key is " " + the lowercased term, so a
    /// term only starts at a word boundary.
    private let prefixes: Set<String>
    /// Matches under way: term prefixes the emitted text currently ends with.
    private var active: [String] = []
    /// Returns nil when no term survives normalization.
    init?(
        terms: [String],
        vocabulary: [String],
        blankToken: Int,
        config: NemotronASRTermBoostConfig
    ) {
        var prefixes = Set<String>()
        for term in terms {
            let words = term.lowercased().split(whereSeparator: { $0.isWhitespace })
            guard !words.isEmpty else { continue }
            let key = " " + words.joined(separator: " ")
            var prefix = ""
            for character in key {
                prefix.append(character)
                prefixes.insert(prefix)
            }
        }
        guard !prefixes.isEmpty else { return nil }
        self.prefixes = prefixes
        self.config = config
        self.pieceText = vocabulary.indices.map { id in
            guard id != blankToken, !NemotronASRTokenizer.isSpecialToken(id, vocabulary: vocabulary) else {
                return nil
            }
            return vocabulary[id].replacingOccurrences(of: "▁", with: " ").lowercased()
        }
    }

    /// The token to emit instead of `greedy`, the unbiased argmax of `logits`.
    func choose(logits: [Float], greedy: Int) -> Int {
        guard greedy < pieceText.count, pieceText[greedy] != nil else { return greedy }
        let floor = logits[greedy] - config.margin
        var best = greedy
        var bestScore = logits[greedy] + bonus(greedy)
        for id in 0..<min(logits.count, pieceText.count) where id != greedy && logits[id] >= floor {
            let score = logits[id] + bonus(id)
            if score > bestScore {
                best = id
                bestScore = score
            }
        }
        return best
    }

    /// Advance the matches with an emitted token.
    func accept(_ token: Int) {
        guard token < pieceText.count, let piece = pieceText[token] else {
            active = []
            return
        }
        var next: [String] = []
        for match in active {
            let extended = match + piece
            if prefixes.contains(extended) { next.append(extended) }
        }
        if piece.hasPrefix(" "), prefixes.contains(piece), !next.contains(piece) {
            next.append(piece)
        }
        active = next
    }

    private func bonus(_ id: Int) -> Float {
        guard let piece = pieceText[id], piece.contains(where: { $0 != " " }) else { return 0 }
        var starts = piece.hasPrefix(" ") && prefixes.contains(piece)
        for match in active where prefixes.contains(match + piece) {
            // A bare "▁" emitted before the term's first letters starts it too.
            if match.contains(where: { $0 != " " }) { return config.continuationBoost }
            starts = true
        }
        return starts ? config.firstTokenBoost : 0
    }
}
