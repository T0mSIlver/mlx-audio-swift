import Foundation
import Testing

/// Every download in Sources must go to the cache its caller passed in (#242, #279).
/// `resolveOrDownloadModel`, `HubClient` and `HubCache` all default to `.default`, so a
/// call that forgets `cache:` still compiles and silently writes to ~/.cache instead.
/// This reads the sources rather than running each loader, which would need a fake
/// repo per model.
struct HubCacheForwardingTests {
    /// Loaders that take no cache parameter yet, so they have nothing to forward.
    static let noCacheParameterYet: Set<String> = [
        "MLXAudioLID/Models/EcapaTdnn/EcapaTdnnLID.swift",
        "MLXAudioLID/Models/Wav2Vec2/Wav2Vec2LID.swift",
        "MLXAudioVAD/Models/SileroVAD/SileroVAD.swift",
        "MLXAudioVAD/Models/SmartTurn/SmartTurn.swift",
    ]

    static let sourcesDirectory = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("Sources")

    /// The text of each call to `name(`, from the name to its closing parenthesis.
    static func calls(to name: String, in source: String) -> [(line: Int, text: String)] {
        var result: [(Int, String)] = []
        var searchStart = source.startIndex
        while let match = source.range(of: name + "(", range: searchStart..<source.endIndex) {
            searchStart = match.upperBound
            let before = source[..<match.lowerBound]
            // Skip declarations, and longer names that end in `name`.
            if before.hasSuffix("func ") || before.last.map({ $0.isLetter || $0.isNumber }) == true {
                continue
            }
            var depth = 1
            var index = match.upperBound
            while depth > 0, index < source.endIndex {
                if source[index] == "(" { depth += 1 }
                if source[index] == ")" { depth -= 1 }
                index = source.index(after: index)
            }
            let line = source[..<match.lowerBound].filter { $0 == "\n" }.count + 1
            result.append((line, String(source[match.lowerBound..<index])))
        }
        return result
    }

    @Test func everyDownloadForwardsItsCache() throws {
        let files = FileManager.default.enumerator(at: Self.sourcesDirectory, includingPropertiesForKeys: nil)!
            .compactMap { $0 as? URL }
            .filter { $0.pathExtension == "swift" }
        #expect(!files.isEmpty, "no sources found at \(Self.sourcesDirectory.path)")

        var offenders: [String] = []
        for file in files {
            let relative = String(file.path.dropFirst(Self.sourcesDirectory.path.count + 1))
            if Self.noCacheParameterYet.contains(relative) { continue }
            let source = try String(contentsOf: file, encoding: .utf8)

            for call in Self.calls(to: "resolveOrDownloadModel", in: source)
            where !call.text.contains("cache:") && !call.text.contains("client:") {
                offenders.append("\(relative):\(call.line) resolveOrDownloadModel without cache:")
            }
            for call in Self.calls(to: "HubClient", in: source) where !call.text.contains("cache:") {
                offenders.append("\(relative):\(call.line) HubClient without cache:")
            }
            for (number, line) in source.split(separator: "\n", omittingEmptySubsequences: false).enumerated()
            where line.contains("HubCache.default") && !line.contains("HubCache = HubCache.default") {
                offenders.append("\(relative):\(number + 1) HubCache.default")
            }
        }
        #expect(offenders.isEmpty, "\(offenders.sorted().joined(separator: "\n"))")
    }
}
