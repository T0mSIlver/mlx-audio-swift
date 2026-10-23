import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXAudioSTT
import Testing

/// CI-only check for Blaizzy#259: load SenseVoiceSmall into an empty cache without modelType,
/// list what landed on disk, and transcribe a LibriSpeech clip.
struct SenseVoice259Check {
    static let reference = "HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOUR FATTENED SAUCE"

    static func run(arm: String, modelType: String?) async throws {
        guard ProcessInfo.processInfo.environment["MLXAUDIO_CHECK_259"] == "1" else { return }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("hub-\(UUID().uuidString)")
        let cache = HubCache(cacheDirectory: root)
        let repo = "mlx-community/SenseVoiceSmall"
        let model: any STTGenerationModel = if let modelType {
            try await STT.loadModel(modelRepo: repo, modelType: modelType, cache: cache)
        } else {
            try await STT.loadModel(modelRepo: repo, cache: cache)
        }
        let dir = root.appendingPathComponent("mlx-audio/mlx-community_SenseVoiceSmall")
        let files = (try? FileManager.default.contentsOfDirectory(atPath: dir.path).sorted()) ?? []
        print("CHECK259 \(arm) files: \(files)")
        let url = Bundle.module.url(forResource: "ls1089", withExtension: "wav", subdirectory: "media")!
        let (_, audio) = try loadAudioArray(from: url, sampleRate: 16000)
        let out = model.generate(audio: audio, generationParameters: model.defaultGenerationParameters)
        print("CHECK259 \(arm) text: \(out.text)")
        print("CHECK259 \(arm) reference: \(reference)")
    }

    @Test func withoutModelType() async throws { try await Self.run(arm: "no_model_type", modelType: nil) }
    @Test func withModelType() async throws { try await Self.run(arm: "model_type_sensevoice", modelType: "sensevoice") }
}
