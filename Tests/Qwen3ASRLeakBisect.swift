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

    @Test func bisect() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_LEAK_283"] == "1" else { return }
        for repo in ["mlx-community/Qwen3-ASR-0.6B-4bit", "mlx-community/Qwen3-ASR-1.7B-8bit"] {
            let dir = try await ModelUtils.resolveOrDownloadModel(
                repoID: Repo.ID(rawValue: repo)!, requiredExtension: "safetensors")
            try Qwen3ASRModel.generateTokenizerJSONIfMissing(in: dir)
            let config = try JSONDecoder().decode(
                Qwen3ASRConfig.self, from: Data(contentsOf: dir.appendingPathComponent("config.json")))
            let weightFiles = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "safetensors" }
            print("LEAK283 repo \(repo)")

            try await Self.phase("weights_only") {
                for file in weightFiles { _ = try MLX.loadArrays(url: file) }
            }
            try await Self.phase("tokenizer_only") {
                let tokenizer = try await AutoTokenizer.from(modelFolder: dir)
                _ = tokenizer.encode(text: "hello")
            }
            try await Self.phase("model_init_only") {
                let model = Qwen3ASRModel(config)
                eval(model)
            }
            try await Self.phase("full_load") {
                weak var weakModel: Qwen3ASRModel?
                do {
                    let model = try await Qwen3ASRModel.fromModelDirectory(dir)
                    weakModel = model
                }
                #expect(weakModel == nil)
            }
        }
    }
}
