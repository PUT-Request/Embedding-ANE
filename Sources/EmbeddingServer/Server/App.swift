import Hummingbird
import Logging
import Foundation
import EmbeddingPipeline

typealias AppRouter = Router<BasicRequestContext>

func buildApp(_ config: ServerConfig) async throws -> Application<AppRouter.Responder> {
    print("[Server] Loading tokenizer...")
    let tokenizer = TokenizerWrapper()
    let bundleDir = (config.modelPath as NSString).deletingLastPathComponent
    let localHF = (bundleDir as NSString).appendingPathComponent("hf_model")
    if FileManager.default.fileExists(
        atPath: (localHF as NSString).appendingPathComponent("tokenizer.json")) {
        do {
            try await tokenizer.loadFromFolder(localHF)
        } catch {
            print("[Server] Warning: Could not load local tokenizer: \(error)")
            print("[Server] Falling back to Hub tokenizer")
            try? await tokenizer.loadFromHub()
        }
    } else {
        print("[Server] No local hf_model dir at \(localHF); loading tokenizer from Hub")
        try? await tokenizer.loadFromHub()
    }

    print("[Server] Loading CoreML model from: \(config.modelPath)")
    let model = try EmbeddingModel(
        modelPath: config.modelPath,
        computeUnits: config.computeUnitConfig
    )

    let pipeline = EmbeddingPipeline(
        model: model,
        tokenizer: tokenizer,
        defaultDimensions: config.defaultDimensions,
        maxBatchSize: config.maxBatchSize
    )

    let routes = Routes(pipeline: pipeline, config: config)

    var logger = Logger(label: "embedding-ane")
    logger.logLevel = .info

    return Application(
        router: routes.app,
        configuration: .init(address: .hostname(config.host, port: config.port)),
        logger: logger
    )
}
