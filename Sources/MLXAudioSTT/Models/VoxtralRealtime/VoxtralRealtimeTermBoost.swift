import Foundation

// Decode-time term boosting for the streaming Voxtral Realtime decoder.
//
// The decoder has no text slot: its prefix is BOS plus streaming pads, and each
// step consumes an audio embedding plus the last token. So a term list cannot be
// a prompt. Instead, while decoding, a token that starts or continues a listed
// term gets a bonus on its logit before the argmax. Nothing is retrained.
//
// The terms become a token trie with the model's own tokenizer, in the variants
// a speaker's words come out as: with a leading space, capitalized, and with
// hyphens as spaces ("mlx-audio-swift" said as "mlx audio swift").
//
// Rules that keep the boost from inventing words:
//   * Nothing is boosted unless the unbiased top token is a text token. Most steps
//     emit [STREAMING_PAD] or [STREAMING_WORD]; the boost never turns one into
//     text, so it cannot write ahead of the audio.
//   * A token competes only when its logit is within `margin` of the top logit.
//   * A term's first token gets a smaller bonus than the next token of a match
//     under way: output is append-only, so a wrong first token is the costly one.
//   * [STREAMING_WORD] ends a match unless the term's next token starts a new word
//     ("Claude Code"); any other non-text token ends every match; padding does
//     neither, since a word's tokens can be spread over several steps.
//
// The same rules, measured on vLLM over 260 term-recall cases, gained 19 term
// occurrences and lost none (localvoxtral #316). On this decoder, 4-bit, over 90
// cases (en and fr), the defaults below took term recall from 73.5% to 78.3% with
// no false insertion and word error rate flat, and an unrelated list changed
// nothing; 1.5/4/3 gained half as much, 3/5/6 started inserting terms.

/// Bonus sizes for `VoxtralRealtimeStreamSession.setBoostTerms(_:config:)`, in logits.
public struct VoxtralRealtimeTermBoostConfig: Sendable, Equatable {
    /// Added to a token that starts a listed term at a word boundary.
    public var firstTokenBoost: Float
    /// Added to a token that continues a term whose earlier tokens were emitted.
    public var continuationBoost: Float
    /// Only tokens whose logit is within this distance of the unbiased top logit
    /// can take its place.
    public var margin: Float

    public init(firstTokenBoost: Float = 2.5, continuationBoost: Float = 4.0, margin: Float = 4.5) {
        self.firstTokenBoost = firstTokenBoost
        self.continuationBoost = continuationBoost
        self.margin = margin
    }
}

final class VoxtralRealtimeTermBooster {
    /// Tekken's [STREAMING_WORD]: the decoder emits it between words.
    static let streamingWordTokenId = 33

    final class Node {
        var children: [Int: Node] = [:]
        /// A child token begins with a space, so a match here survives a word boundary.
        var startsWord = false
    }

    let config: VoxtralRealtimeTermBoostConfig
    private let root = Node()
    private var live: [Node] = []
    private let padToken: Int
    private let wordToken: Int
    private let firstTextToken: Int

    /// Nil when no term yields a token.
    init?(
        terms: [String],
        encode: (String) -> [Int],
        startsWithSpace: (Int) -> Bool,
        padToken: Int,
        wordToken: Int = VoxtralRealtimeTermBooster.streamingWordTokenId,
        firstTextToken: Int,
        config: VoxtralRealtimeTermBoostConfig
    ) {
        self.config = config
        self.padToken = padToken
        self.wordToken = wordToken
        self.firstTextToken = firstTextToken
        for term in terms {
            for text in Self.variants(term) {
                var node = root
                for token in encode(text) where token >= firstTextToken {
                    if let child = node.children[token] {
                        node = child
                    } else {
                        let child = Node()
                        node.children[token] = child
                        if startsWithSpace(token) { node.startsWord = true }
                        node = child
                    }
                }
            }
        }
        guard !root.children.isEmpty else { return nil }
    }

    /// " term", " Term", and the same with hyphens as spaces.
    static func variants(_ term: String) -> Set<String> {
        let trimmed = term.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        guard !trimmed.isEmpty else { return [] }
        var spellings: Set<String> = [trimmed, trimmed.replacingOccurrences(of: "-", with: " ")]
        for spelling in spellings {
            spellings.insert(spelling.prefix(1).uppercased() + spelling.dropFirst())
        }
        return Set(spellings.map { " " + $0 })
    }

    /// The token to emit instead of `greedy`, the unbiased argmax. `logits` returns
    /// the logits of the ids it is given, in order; it is called only when a
    /// candidate exists, so most steps read nothing back.
    func choose(greedy: Int, logits: ([Int]) -> [Float]) -> Int {
        guard greedy >= firstTextToken else { return greedy }
        let bonuses = currentBonuses()
        guard !bonuses.isEmpty else { return greedy }
        let ids = Array(bonuses.keys)
        let values = logits(ids + [greedy])
        guard values.count == ids.count + 1 else { return greedy }
        let top = values[ids.count]
        var best = greedy
        var bestScore = top + (bonuses[greedy] ?? 0)
        for (id, value) in zip(ids, values) where id != greedy && top - value <= config.margin {
            let score = value + bonuses[id, default: 0]
            if score > bestScore {
                best = id
                bestScore = score
            }
        }
        return best
    }

    /// Advance the matches with an emitted token.
    func accept(_ token: Int) {
        if token == padToken { return }
        if token == wordToken {
            live = live.filter(\.startsWord)
            return
        }
        guard token >= firstTextToken else {
            live = []
            return
        }
        var next = live.compactMap { $0.children[token] }
        if let start = root.children[token] { next.append(start) }
        live = next
    }

    /// First-token bonuses for every term start, continuation bonuses for the
    /// next tokens of live matches. A negative setting counts as zero: it would
    /// otherwise let a token outside every term win.
    private func currentBonuses() -> [Int: Float] {
        let first = max(config.firstTokenBoost, 0)
        let continuation = max(config.continuationBoost, 0)
        var bonuses: [Int: Float] = [:]
        for token in root.children.keys { bonuses[token] = first }
        for node in live {
            for token in node.children.keys {
                bonuses[token] = max(bonuses[token] ?? 0, continuation)
            }
        }
        return bonuses
    }
}
