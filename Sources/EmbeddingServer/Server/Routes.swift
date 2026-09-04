import Hummingbird
import NIOCore
import EmbeddingPipeline

struct Routes {
    let app: AppRouter
    let pipeline: EmbeddingPipeline
    let config: ServerConfig

    init(pipeline: EmbeddingPipeline, config: ServerConfig) {
        self.app = AppRouter()
        self.pipeline = pipeline
        self.config = config

        app.post("/v1/embeddings") { [self] request, context -> EmbeddingResponse in
            let embeddingRequest = try await request.decode(as: EmbeddingRequest.self, context: context)

            guard embeddingRequest.model == config.modelName || embeddingRequest.model == "auto" else {
                throw HTTPError(
                    .badRequest,
                    message: "Model '\(embeddingRequest.model)' not found. Available: \(config.modelName)"
                )
            }

            let texts = embeddingRequest.input.texts
            let dimensions = embeddingRequest.dimensions

            guard !texts.isEmpty else {
                throw HTTPError(.badRequest, message: "Input cannot be empty")
            }

            guard texts.count <= config.maxBatchSize else {
                throw HTTPError(
                    .badRequest,
                    message: "Batch size \(texts.count) exceeds maximum \(config.maxBatchSize)"
                )
            }

            let results = try pipeline.embedBatch(texts, dimensions: dimensions)
            let totalTokens = results.reduce(0) { $0 + $1.tokenCount }

            let data = results.enumerated().map { index, result in
                EmbeddingData(embedding: result.embedding, index: index)
            }

            return EmbeddingResponse(
                data: data,
                model: config.modelName,
                usage: Usage(promptTokens: totalTokens)
            )
        }

        app.get("/v1/models") { _, _ in
            ModelListResponse(data: [ModelInfo(id: config.modelName)])
        }

        app.get("/health") { _, _ in
            ["status": "ok", "model": config.modelName] as [String: String]
        }
    }
}
