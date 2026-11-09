import Foundation

// Hotword boosting for the greedy RNN-T: NeMo's word boosting (GPU-PB).
//
// This follows NeMo's GPU phrase-boosting tree in shallow fusion
// (nemo/collections/asr/parts/context_biasing/boosting_graph_batched.py and
// TurboBias, arXiv:2508.07014) as it runs in greedy label-looping decoding:
//
//   * The unbiased argmax decides blank or not. On a blank frame nothing changes;
//     otherwise every non-blank token gets `alpha` times its tree score added to
//     its logit and the argmax is taken again over non-blank tokens.
//   * A token that starts a phrase scores `contextScore`; the token at depth
//     d >= 1 of a match scores `contextScore * depthScaling + ln(d + 1)`.
//   * A token that breaks a partial match takes the failure arc, whose score gives
//     back what the match had collected, so a wrong start costs its own bonus. A
//     match that completed a phrase backs off at no cost. A token outside every
//     phrase scores 0 (NeMo's `unk_score`).
//
// One difference: NeMo builds its tree from the tokenizer's ids for each phrase.
// Here matching runs on the lowercased text a token appends ("▁" as a space), so
// every tokenization and casing of a hotword is boosted, as NeMo's
// `case_insensitive` mode intends, without needing a SentencePiece encoder. A
// phrase starts only at a word boundary, and a bare "▁" piece moves a match along
// without adding depth or score.

/// Weights for hotword boosting in `NemotronASRModel.makeStreamSession(...)`.
public struct NemotronASRHotwordsConfig: Sendable, Equatable {
    /// NeMo's `boosting_tree_alpha`: the weight of the boosting tree's score.
    public var alpha: Float
    /// NeMo's `context_score`: the score of a phrase's first token.
    public var contextScore: Float
    /// NeMo's `depth_scaling`; NeMo recommends 2.0 for RNN-T.
    public var depthScaling: Float

    public init(alpha: Float = 1.0, contextScore: Float = 1.0, depthScaling: Float = 2.0) {
        self.alpha = alpha
        self.contextScore = contextScore
        self.depthScaling = depthScaling
    }
}

final class NemotronASRHotwordBooster {
    private struct Match {
        var text: String  // the phrase prefix the emitted text ends with
        var depth: Int  // tokens it took, bare "▁" pieces excluded
        var score: Float  // tree score collected along the way, alpha included
    }

    private static let root = Match(text: "", depth: 0, score: 0)

    let config: NemotronASRHotwordsConfig
    private let blankToken: Int
    private let keys: Set<String>
    /// Token ids by the lowercased text they append, specials and blank left out.
    private let pieceIDs: [String: [Int]]
    private let emittable: Set<Int>
    /// Tokens whose text has a letter, digit or mark, not only spaces.
    private let textPieces: Set<Int>
    private let maxPieceLength: Int
    private var extensionsCache: [String: [(token: Int, text: String)]] = [:]
    /// Matches under way, longest first: NeMo's failure chain. The root is implicit.
    private var chain: [Match]
    private var cachedScores: (unmatched: Float, tokens: [Int: Float])?

    /// Returns nil when no hotword survives normalization.
    init?(hotwords: [String], vocabulary: [String], blankToken: Int, config: NemotronASRHotwordsConfig) {
        var keys = Set<String>()
        for hotword in hotwords {
            let words = hotword.lowercased().split(whereSeparator: { $0.isWhitespace })
            if !words.isEmpty { keys.insert(" " + words.joined(separator: " ")) }
        }
        guard !keys.isEmpty else { return nil }
        self.keys = keys
        self.config = config
        self.blankToken = blankToken
        var pieceIDs: [String: [Int]] = [:]
        var textPieces = Set<Int>()
        for (id, piece) in vocabulary.enumerated()
        where id != blankToken && !NemotronASRTokenizer.isSpecialToken(id, vocabulary: vocabulary) {
            let text = piece.replacingOccurrences(of: "▁", with: " ").lowercased()
            guard !text.isEmpty else { continue }
            pieceIDs[text, default: []].append(id)
            if text.contains(where: { $0 != " " }) { textPieces.insert(id) }
        }
        self.pieceIDs = pieceIDs
        self.emittable = Set(pieceIDs.values.joined())
        self.textPieces = textPieces
        self.maxPieceLength = pieceIDs.keys.map(\.count).max() ?? 0
        // The start of a transcript is a word boundary: a piece with no leading "▁"
        // still starts a word there.
        self.chain = [Match(text: " ", depth: 0, score: 0)]
    }

    private func arc(_ depth: Int) -> Float {
        let score = depth == 0
            ? config.contextScore
            : config.contextScore * config.depthScaling + Float(log(Double(depth + 1)))
        return config.alpha * score
    }

    /// Tokens whose text extends `text` to a phrase prefix, with that prefix.
    private func extensions(_ text: String) -> [(token: Int, text: String)] {
        if let cached = extensionsCache[text] { return cached }
        var found: [Int: String] = [:]
        for key in keys where key.count > text.count && key.hasPrefix(text) {
            let rest = key.dropFirst(text.count)
            var end = rest.startIndex
            for _ in 0..<min(rest.count, maxPieceLength) {
                end = rest.index(after: end)
                let piece = String(rest[rest.startIndex..<end])
                for token in pieceIDs[piece] ?? [] where found[token] == nil {
                    found[token] = text + piece
                }
            }
        }
        let result = found.map { (token: $0.key, text: $0.value) }
        extensionsCache[text] = result
        return result
    }

    private func step(_ match: Match, _ token: Int, _ text: String) -> Match {
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
        guard greedy != blankToken else { return greedy }
        let (unmatched, tokens) = scores()
        var best = greedy
        var bestScore = -Float.infinity
        for (token, logit) in logits.enumerated() where token != blankToken {
            let score = logit + (tokens[token] ?? unmatched)
            if score > bestScore {
                best = token
                bestScore = score
            }
        }
        return best
    }

    /// Advance the matches with an emitted token. Special tokens, such as a
    /// language tag, leave them as they are.
    func accept(_ token: Int) {
        guard emittable.contains(token) else { return }
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
