import Foundation

public enum EmbeddingPipelineError: Error, CustomStringConvertible {
    case modelNotLoaded
    case tokenizerNotLoaded
    case inferenceFailed(String)

    public var description: String {
        switch self {
        case .modelNotLoaded: return "Embedding model not loaded"
        case .tokenizerNotLoaded: return "Tokenizer not loaded"
        case .inferenceFailed(let msg): return "Inference failed: \(msg)"
        }
    }
}

public final class EmbeddingPipeline {
    private let model: EmbeddingModel
    private let tokenizer: TokenizerWrapper
    private let defaultDimensions: Int
    private let maxBatchSize: Int

    public init(model: EmbeddingModel, tokenizer: TokenizerWrapper, defaultDimensions: Int = 768, maxBatchSize: Int = 32) {
        self.model = model
        self.tokenizer = tokenizer
        self.defaultDimensions = defaultDimensions
        self.maxBatchSize = maxBatchSize
    }

    public func embed(_ text: String, dimensions: Int? = nil) throws -> EmbeddingResult {
        let targetDim = dimensions ?? defaultDimensions
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !trimmed.isEmpty else {
            return EmbeddingResult(embedding: Array(repeating: Float16(0), count: targetDim), tokenCount: 0)
        }

        let rawEncoded = tokenizer.encode(trimmed, maxLength: 2048, padding: false)
        let bucket = model.selectBucket(for: rawEncoded.tokenCount)
        let encoded = padToLength(rawEncoded, length: bucket)

        let embedding = try model.predict(
            inputIds: encoded.inputIds,
            attentionMask: encoded.attentionMask,
            seqLength: bucket
        )

        let truncated = truncateDimensions(embedding, to: targetDim)
        let normalized = l2Normalize(truncated)

        return EmbeddingResult(embedding: normalized, tokenCount: encoded.tokenCount)
    }

    public func embedBatch(_ texts: [String], dimensions: Int? = nil) throws -> [EmbeddingResult] {
        let targetDim = dimensions ?? defaultDimensions
        var results: [EmbeddingResult] = []
        results.reserveCapacity(texts.count)

        for text in texts {
            let result = try embed(text, dimensions: targetDim)
            results.append(result)
        }

        return results
    }

    public func embedBatchParallel(_ texts: [String], dimensions: Int? = nil) throws -> [EmbeddingResult] {
        let targetDim = dimensions ?? defaultDimensions
        let batchSize = min(texts.count, maxBatchSize)
        var results = [EmbeddingResult?](repeating: nil, count: texts.count)

        var chunks: [[(index: Int, text: String)]] = []
        for i in stride(from: 0, to: texts.count, by: batchSize) {
            let end = min(i + batchSize, texts.count)
            let chunk = Array(i..<end).map { (index: $0, text: texts[$0]) }
            chunks.append(chunk)
        }

        for chunk in chunks {
            for item in chunk {
                let result = try embed(item.text, dimensions: targetDim)
                results[item.index] = result
            }
        }

        return results.compactMap { $0 }
    }

    private func truncateDimensions(_ vector: [Float16], to targetDim: Int) -> [Float16] {
        guard targetDim < vector.count else { return vector }
        return Array(vector.prefix(targetDim))
    }

    private func padToLength(_ encoded: TokenizerResult, length: Int) -> TokenizerResult {
        var ids = encoded.inputIds
        var mask = encoded.attentionMask
        if ids.count > length {
            ids = Array(ids.prefix(length))
            mask = Array(mask.prefix(length))
        }
        while ids.count < length {
            ids.append(0)
            mask.append(0)
        }
        return TokenizerResult(inputIds: ids, attentionMask: mask, tokenCount: encoded.tokenCount)
    }

    private func l2Normalize(_ vector: [Float16]) -> [Float16] {
        var normSq: Float16 = 0
        for x in vector { normSq += x * x }
        let norm = sqrt(normSq)
        guard norm > 0 else { return vector }
        var result = vector
        for i in result.indices { result[i] /= norm }
        return result
    }
}

public struct EmbeddingResult {
    public let embedding: [Float16]
    public let tokenCount: Int
}
