import Foundation

// Hotword boosting for the greedy RNN-T: NeMo's word boosting (GPU-PB).
//
// This follows NeMo's GPU phrase-boosting tree in shallow fusion
// (nemo/collections/asr/parts/context_biasing/boosting_graph_batched.py and
// TurboBias, arXiv:2508.07014) as it runs in greedy label-looping decoding:
//
//   * The unbiased argmax decides blank or not. On a blank frame nothing changes;
//     otherwise every non-blank token gets `alpha` times its tree score added to
//     its logit and the argmax is taken again over non-blank tokens. A frame whose
//     argmax is a special token, such as a language tag, is left alone too, and
//     special tokens never take a frame from a text token.
//   * A token that starts a phrase scores NeMo's `context_score` (1.0); the token
//     at depth d >= 1 of a match scores `context_score * depth_scaling + ln(d + 1)`,
//     with the `depth_scaling` NeMo recommends for RNN-T (2.0).
//   * A token that breaks a partial match takes the failure arc, whose score gives
//     back what the match had collected, so a wrong start costs its own bonus. A
//     match that completed a phrase backs off at no cost. A token outside every
//     phrase scores 0 (NeMo's `unk_score`).
//
// Where this differs: NeMo builds its tree from one tokenization of each phrase.
// There is no SentencePiece encoder here, so matching runs on the NFKC-normalized,
// case-folded text each token appends ("▁" as a space), as Unicode scalars, the
// same units as the Python port. Every tokenization and casing of a hotword is
// boosted, and depth counts the tokens actually emitted, so a spelling in more
// pieces collects more score than NeMo's single path would. A phrase starts only
// at a word boundary (the start of the transcript counts), a bare "▁" moves a
// match along without depth or score, and special tokens leave the matches alone.

/// NeMo's `boosting_tree_alpha` defaults to 0 (off) and is tuned per dataset.
public let nemotronASRDefaultHotwordsAlpha: Float = 1.0

final class NemotronASRHotwordBooster {
    private typealias Text = [Unicode.Scalar]

    private struct Match {
        var text: Text  // the phrase prefix the emitted text ends with
        var depth: Int  // tokens it took, bare "▁" pieces excluded
        var score: Float  // tree score collected along the way, alpha included
    }

    private static let contextScore: Float = 1.0
    private static let depthScaling: Float = 2.0
    private static let space: Unicode.Scalar = " "
    private static let root = Match(text: [], depth: 0, score: 0)

    let alpha: Float
    private let keys: Set<Text>
    /// Token ids by the lowercased text they append, specials and blank left out.
    private let pieceIDs: [Text: [Int]]
    /// Whether a token can take a frame: false for blank and special tokens.
    private let emittable: [Bool]
    /// Tokens whose text has something besides spaces.
    private let textPieces: Set<Int>
    private let maxPieceLength: Int
    private var extensionsCache: [Text: [(token: Int, text: Text)]] = [:]
    /// Matches under way, longest first: NeMo's failure chain. The root is implicit.
    private var chain: [Match]
    private var cachedScores: (unmatched: Float, tokens: [Int: Float])?

    /// Lowercase one scalar at a time, with final sigma folded to sigma. A hotword
    /// and the pieces that spell it must fold the same way, so casing cannot depend
    /// on context: "ΟΣ".lowercased() ends in "ς" but the piece "Σ" lowers to "σ".
    static func fold(_ string: String) -> String {
        string.unicodeScalars.map { String($0).lowercased() }.joined().replacingOccurrences(of: "ς", with: "σ")
    }

    /// Returns nil when no hotword survives normalization.
    init?(hotwords: [String], vocabulary: [String], blankToken: Int, alpha: Float) {
        var keys = Set<Text>()
        for hotword in hotwords {
            let words = Self.fold(hotword.precomposedStringWithCompatibilityMapping).split(whereSeparator: { $0.isWhitespace })
            if !words.isEmpty { keys.insert(Array((" " + words.joined(separator: " ")).unicodeScalars)) }
        }
        guard !keys.isEmpty else { return nil }
        self.keys = keys
        self.alpha = alpha
        var pieceIDs: [Text: [Int]] = [:]
        var textPieces = Set<Int>()
        var emittable = [Bool](repeating: false, count: vocabulary.count + 1)
        for (id, piece) in vocabulary.enumerated()
        where id != blankToken && !NemotronASRTokenizer.isSpecialToken(id, vocabulary: vocabulary) {
            let text = Array(Self.fold(piece.replacingOccurrences(of: "▁", with: " ")).unicodeScalars)
            guard !text.isEmpty else { continue }
            pieceIDs[text, default: []].append(id)
            emittable[id] = true
            if text.contains(where: { $0 != Self.space }) { textPieces.insert(id) }
        }
        self.pieceIDs = pieceIDs
        self.emittable = emittable
        self.textPieces = textPieces
        self.maxPieceLength = pieceIDs.keys.map(\.count).max() ?? 0
        // The start of a transcript is a word boundary: a piece with no leading "▁"
        // still starts a word there.
        self.chain = [Match(text: [Self.space], depth: 0, score: 0)]
    }

    private func arc(_ depth: Int) -> Float {
        let score = depth == 0
            ? Self.contextScore
            : Self.contextScore * Self.depthScaling + Float(log(Double(depth + 1)))
        return alpha * score
    }

    /// Tokens whose text extends `text` to a phrase prefix, with that prefix.
    private func extensions(_ text: Text) -> [(token: Int, text: Text)] {
        if let cached = extensionsCache[text] { return cached }
        var found: [Int: Text] = [:]
        for key in keys where key.count > text.count && key.starts(with: text) {
            let rest = key[text.count...]
            for length in stride(from: 1, through: min(rest.count, maxPieceLength), by: 1) {
                let piece = Text(rest.prefix(length))
                for token in pieceIDs[piece] ?? [] where found[token] == nil {
                    found[token] = text + piece
                }
            }
        }
        let result = found.map { (token: $0.key, text: $0.value) }
        extensionsCache[text] = result
        return result
    }

    private func step(_ match: Match, _ token: Int, _ text: Text) -> Match {
        guard textPieces.contains(token) else {
            return Match(text: text, depth: match.depth, score: match.score)
        }
        return Match(text: text, depth: match.depth + 1, score: match.score + arc(match.depth))
    }

    /// The score of every token outside the matches, and of each token in one.
    func scores() -> (unmatched: Float, tokens: [Int: Float]) {
        if let cachedScores { return cachedScores }
        let full = chain + [Self.root]
        var tokens: [Int: Float] = [:]
        var backoff: Float = 0
        for (index, match) in full.enumerated() {
            for (token, text) in extensions(match.text) where tokens[token] == nil {
                tokens[token] = backoff + step(match, token, text).score - match.score
            }
            if index + 1 < full.count, !keys.contains(match.text) {
                backoff += full[index + 1].score - match.score
            }
        }
        cachedScores = (backoff, tokens)
        return (backoff, tokens)
    }

    /// The token to emit instead of `greedy`, the unbiased argmax of `logits`.
    func choose(logits: [Float], greedy: Int) -> Int {
        guard greedy < emittable.count, emittable[greedy] else { return greedy }
        let (unmatched, tokens) = scores()
        var biased = logits
        for index in biased.indices {
            biased[index] = index < emittable.count && emittable[index] ? logits[index] + unmatched : -.infinity
        }
        for (token, score) in tokens where token < biased.count {
            biased[token] = logits[token] + score
        }
        // The first maximum, as numpy's argmax takes it.
        var best = greedy
        var bestScore = -Float.infinity
        for (index, value) in biased.enumerated() where value > bestScore {
            best = index
            bestScore = value
        }
        return best
    }

    /// Advance the matches with an emitted token. Special tokens, such as a
    /// language tag, leave them as they are.
    func accept(_ token: Int) {
        guard token < emittable.count, emittable[token] else { return }
        var next: [Match] = []
        for match in chain + [Self.root] {
            for (candidate, text) in extensions(match.text) where candidate == token {
                next.append(step(match, token, text))
            }
        }
        chain = next.sorted { $0.text.count > $1.text.count }
        cachedScores = nil
    }
}
