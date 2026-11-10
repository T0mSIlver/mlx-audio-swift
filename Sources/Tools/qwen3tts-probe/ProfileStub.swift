// The per-step timers live only on the baseline bench branch.
enum Qwen3TTSProfile {
    nonisolated(unsafe) static var splitEval = false
    static func reset() {}
    static func report(_ tag: String) -> String { "PROFILE \(tag) not instrumented" }
}
