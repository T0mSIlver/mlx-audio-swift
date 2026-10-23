import Foundation
import Testing

/// A loader that takes a `cache:` must hand it to every download it makes (#279, #242).
/// `ModelUtils.resolveOrDownloadModel` defaults `cache` to `.default`, so a call that omits it
/// still compiles and quietly downloads somewhere else. These checks read the sources instead.
@Suite("Downloads use the caller's cache")
struct CacheForwardingTests {
    /// Calls that can't forward a cache yet, because their entry point takes none.
    /// The count pins them: a new call in one of these files still fails.
    static let allowed: [String: Int] = [
        "MLXAudioLID/Models/EcapaTdnn/EcapaTdnnLID.swift": 1,
        "MLXAudioLID/Models/Wav2Vec2/Wav2Vec2LID.swift": 1,
        "MLXAudioVAD/Models/SileroVAD/SileroVAD.swift": 1,
        "MLXAudioVAD/Models/SmartTurn/SmartTurn.swift": 1,
        // hf:// voice files resolved at generation time, after loading.
        "MLXAudioTTS/Models/PocketTTS/PocketTTSModel.swift": 2,
    ]

    static let sourcesURL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Sources")

    /// Each Swift source as (path relative to Sources, text without comment lines).
    static func sources() throws -> [(String, String)] {
        let enumerator = FileManager.default.enumerator(at: sourcesURL, includingPropertiesForKeys: nil)
        var result: [(String, String)] = []
        while let url = enumerator?.nextObject() as? URL {
            guard url.pathExtension == "swift" else { continue }
            let text = try String(contentsOf: url, encoding: .utf8)
                .split(separator: "\n", omittingEmptySubsequences: false)
                .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//") }
                .joined(separator: "\n")
            let path = String(url.standardizedFileURL.path.dropFirst(sourcesURL.standardizedFileURL.path.count + 1))
            result.append((path, text))
        }
        return result
    }

    /// The argument lists of every call to `name`, skipping its declarations.
    static func calls(to name: String, in text: String) -> [Substring] {
        var result: [Substring] = []
        var searchStart = text.startIndex
        while let match = text.range(of: name + "(", range: searchStart..<text.endIndex) {
            searchStart = match.upperBound
            if text[..<match.lowerBound].hasSuffix("func ") { continue }
            var depth = 1
            var index = match.upperBound
            while depth > 0, index < text.endIndex {
                if text[index] == "(" { depth += 1 } else if text[index] == ")" { depth -= 1 }
                index = text.index(after: index)
            }
            result.append(text[match.upperBound..<index])
        }
        return result
    }

    @Test func everyDownloadPassesACache() throws {
        var offenders: [String: Int] = [:]
        for (path, text) in try Self.sources() {
            let calls = Self.calls(to: "resolveOrDownloadModel", in: text)
                + Self.calls(to: "HubClient", in: text)
            let missing = calls.filter { !$0.contains("cache:") }.count
            // An explicit `cache: .default` argument ignores the caller's cache just the same.
            let explicitDefault = text.components(separatedBy: "cache: .default").count - 1
            let uses = missing + explicitDefault + text.components(separatedBy: "HubCache.default.").count - 1
            if uses != Self.allowed[path, default: 0] { offenders[path] = uses }
        }
        #expect(offenders.isEmpty, "Calls that drop the caller's cache, by file (allowed counts in CacheForwardingTests.allowed): \(offenders)")
    }

    @Test func scannerSeesTheSources() throws {
        let sources = try Self.sources()
        let calls = sources.flatMap { Self.calls(to: "resolveOrDownloadModel", in: $0.1) }
        #expect(calls.count > 50)
        #expect(Self.calls(to: "f", in: "func f(a: Int) {}; f(a: g(1), cache: c)") == ["a: g(1), cache: c)"])
    }
}
