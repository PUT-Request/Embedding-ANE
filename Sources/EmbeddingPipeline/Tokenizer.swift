import Foundation
import Tokenizers

public enum TokenizerError: Error, CustomStringConvertible {
    case notInitialized
    case tokenizationFailed(String)

    public var description: String {
        switch self {
        case .notInitialized: return "Tokenizer not initialized"
        case .tokenizationFailed(let msg): return "Tokenization failed: \(msg)"
        }
    }
}

public final class TokenizerWrapper {
    private var tokenizer: Tokenizer?
    private let padTokenId: Int

    public init() {
        self.tokenizer = nil
        self.padTokenId = 0
    }
    public func loadFromHub() async throws {
        do {
            self.tokenizer = try await AutoTokenizer.from(pretrained: "google/embeddinggemma-300m")
            print("[Tokenizer] Loaded from HuggingFace Hub: google/embeddinggemma-300m")
        } catch {
            throw TokenizerError.tokenizationFailed(error.localizedDescription)
        }
    }

    public func loadFromFolder(_ folderPath: String) async throws {
        do {
            let folderURL = URL(fileURLWithPath: folderPath, isDirectory: true)
            guard FileManager.default.fileExists(atPath: folderPath),
                  FileManager.default.fileExists(
                      atPath: folderURL.appendingPathComponent("tokenizer.json").path)
            else {
                throw TokenizerError.tokenizationFailed(
                    "tokenizer.json not found under \(folderPath)")
            }
            self.tokenizer = try await AutoTokenizer.from(modelFolder: folderURL)
            print("[Tokenizer] Loaded from local folder: \(folderPath)")
        } catch {
            throw TokenizerError.tokenizationFailed(error.localizedDescription)
        }
    }

    public func encode(_ text: String, maxLength: Int, padding: Bool = true) -> TokenizerResult {
        guard let tokenizer = tokenizer else {
            return fallbackEncode(text, maxLength: maxLength)
        }

        let ids = tokenizer.encode(text: text, addSpecialTokens: true)
        var inputIds = ids
        var attentionMask = inputIds.map { $0 == padTokenId ? 0 : 1 }

        let originalCount = inputIds.count

        if inputIds.count > maxLength {
            inputIds = Array(inputIds.prefix(maxLength))
            attentionMask = Array(attentionMask.prefix(maxLength))
        }

        if padding && inputIds.count < maxLength {
            let padCount = maxLength - inputIds.count
            inputIds.append(contentsOf: Array(repeating: padTokenId, count: padCount))
            attentionMask.append(contentsOf: Array(repeating: 0, count: padCount))
        }

        return TokenizerResult(
            inputIds: inputIds,
            attentionMask: attentionMask,
            tokenCount: originalCount
        )
    }

    public func encodeBatch(_ texts: [String], maxLength: Int) -> [TokenizerResult] {
        texts.map { encode($0, maxLength: maxLength) }
    }

    private func fallbackEncode(_ text: String, maxLength: Int) -> TokenizerResult {
        let tokens = text.split(separator: " ").map { String($0).hashValue & 0x7FFFFFFF }
        var inputIds = Array(tokens.prefix(maxLength))
        let originalCount = inputIds.count

        if inputIds.count < maxLength {
            inputIds.append(contentsOf: Array(repeating: 0, count: maxLength - inputIds.count))
        }

        var attentionMask = Array(repeating: 1, count: originalCount)
        if attentionMask.count < maxLength {
            attentionMask.append(contentsOf: Array(repeating: 0, count: maxLength - originalCount))
        }

        return TokenizerResult(
            inputIds: inputIds,
            attentionMask: attentionMask,
            tokenCount: originalCount
        )
    }
}

public struct TokenizerResult {
    public let inputIds: [Int]
    public let attentionMask: [Int]
    public let tokenCount: Int
}
