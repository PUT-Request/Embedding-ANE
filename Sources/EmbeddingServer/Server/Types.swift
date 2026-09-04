import Foundation
import Hummingbird

// MARK: - Request Types

struct EmbeddingRequest: Codable {
    let model: String
    let input: EmbeddingInput
    let encodingFormat: EncodingFormat?
    let dimensions: Int?

    enum CodingKeys: String, CodingKey {
        case model, input
        case encodingFormat = "encoding_format"
        case dimensions
    }
}

enum EmbeddingInput: Codable {
    case single(String)
    case batch([String])

    var texts: [String] {
        switch self {
        case .single(let text): return [text]
        case .batch(let texts): return texts
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let single = try? container.decode(String.self) {
            self = .single(single)
        } else if let batch = try? container.decode([String].self) {
            self = .batch(batch)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Expected string or array of strings"
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .single(let text): try container.encode(text)
        case .batch(let texts): try container.encode(texts)
        }
    }
}

enum EncodingFormat: String, Codable {
    case float
    case base64
}

// MARK: - Response Types

struct EmbeddingResponse: ResponseEncodable {
    let object: String
    let data: [EmbeddingData]
    let model: String
    let usage: Usage

    init(data: [EmbeddingData], model: String, usage: Usage) {
        self.object = "list"
        self.data = data
        self.model = model
        self.usage = usage
    }
}

struct EmbeddingData: ResponseEncodable {
    let object: String
    let embedding: [Float16]
    let index: Int

    init(embedding: [Float16], index: Int) {
        self.object = "embedding"
        self.embedding = embedding
        self.index = index
    }
}

struct Usage: ResponseEncodable {
    let promptTokens: Int
    let totalTokens: Int

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case totalTokens = "total_tokens"
    }

    init(promptTokens: Int) {
        self.promptTokens = promptTokens
        self.totalTokens = promptTokens
    }
}

// MARK: - Model List

struct ModelListResponse: ResponseEncodable {
    let object: String
    let data: [ModelInfo]

    init(data: [ModelInfo]) {
        self.object = "list"
        self.data = data
    }
}

struct ModelInfo: ResponseEncodable {
    let id: String
    let object: String
    let ownedBy: String

    enum CodingKeys: String, CodingKey {
        case id, object
        case ownedBy = "owned_by"
    }

    init(id: String) {
        self.id = id
        self.object = "model"
        self.ownedBy = "embedding-ane"
    }
}

// MARK: - Error Types

struct ErrorResponse: Codable {
    let error: ErrorDetail
}

struct ErrorDetail: Codable {
    let message: String
    let type: String
    let code: String?

    init(message: String, type: String = "invalid_request_error", code: String? = nil) {
        self.message = message
        self.type = type
        self.code = code
    }
}
