import Darwin
import Foundation
import HuggingFace
import MLX
import MLXAudioCore
@testable import MLXAudioSTT
import Testing
import Tokenizers

/// CI-only bisect for Blaizzy#283: which part of Qwen3ASRModel.fromModelDirectory keeps process
/// footprint after release. Each phase repeats one step and prints phys_footprint after release.
struct Qwen3ASRLeakBisect {
    static func footprintMiB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
    }

    static func settle() async {
        Memory.clearCache()
        try? await Task.sleep(nanoseconds: 300_000_000)
    }

    static func phase(_ name: String, cycles: Int = 12, _ body: () async throws -> Void) async throws {
        var samples: [Double] = []
        for _ in 0..<cycles {
            try await body()
            await settle()
            samples.append(footprintMiB())
        }
        let tail = Array(samples.suffix(cycles - 3))
        let slope = (tail.last! - tail.first!) / Double(tail.count - 1)
        print("LEAK283 {\"phase\":\"\(name)\",\"first\":\(String(format: "%.1f", samples.first!)),\"last\":\(String(format: "%.1f", samples.last!)),\"per_cycle_mib\":\(String(format: "%.2f", slope)),\"samples\":\(samples.map { String(format: "%.1f", $0) })}")
    }

    static func modelDir() async throws -> URL {
        let dir = try await ModelUtils.resolveOrDownloadModel(
            repoID: Repo.ID(rawValue: "mlx-community/Qwen3-ASR-0.6B-4bit")!, requiredExtension: "safetensors")
        try Qwen3ASRModel.generateTokenizerJSONIfMissing(in: dir)
        return dir
    }

    static var enabled: Bool { ProcessInfo.processInfo.environment["MLXAUDIO_LEAK_283"] == "1" }

    // Each test runs in its own xctest process (one xcodebuild call per test), so phases don't share a heap.

    @Test func jsonParseNoPool() async throws {
        guard Self.enabled else { return }
        let url = try await Self.modelDir().appendingPathComponent("tokenizer.json")
        try await Self.phase("json_parse_no_pool", cycles: 20) {
            _ = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
        }
    }

    @Test func jsonParseInPool() async throws {
        guard Self.enabled else { return }
        let url = try await Self.modelDir().appendingPathComponent("tokenizer.json")
        try await Self.phase("json_parse_in_pool", cycles: 20) {
            try autoreleasepool { _ = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) }
        }
    }

    @Test func tokenizerAsync() async throws {
        guard Self.enabled else { return }
        let dir = try await Self.modelDir()
        try await Self.phase("tokenizer_async", cycles: 20) {
            _ = try await AutoTokenizer.from(modelFolder: dir)
        }
    }

    @Test func tokenizerOnThreadWithPool() async throws {
        guard Self.enabled else { return }
        let dir = try await Self.modelDir()
        try await Self.phase("tokenizer_thread_pool", cycles: 20) {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                let thread = Thread {
                    autoreleasepool {
                        let sem = DispatchSemaphore(value: 0)
                        var failure: Error?
                        // Build on this thread's pool: the async loader only reads files.
                        Task.detached { do { _ = try await AutoTokenizer.from(modelFolder: dir) } catch { failure = error }; sem.signal() }
                        sem.wait()
                        if let failure { cont.resume(throwing: failure) } else { cont.resume() }
                    }
                }
                thread.start()
            }
        }
    }

    @Test func fullLoad() async throws {
        guard Self.enabled else { return }
        let dir = try await Self.modelDir()
        try await Self.phase("full_load", cycles: 20) {
            _ = try await Qwen3ASRModel.fromModelDirectory(dir)
        }
    }
}
