import CoreML
import Foundation

public enum EmbeddingModelError: Error, CustomStringConvertible {
    case modelNotFound(String)
    case compilationFailed(String)
    case predictionFailed(String)
    case invalidInput(String)

    public var description: String {
        switch self {
        case .modelNotFound(let path): return "Model not found at: \(path)"
        case .compilationFailed(let msg): return "Model compilation failed: \(msg)"
        case .predictionFailed(let msg): return "Prediction failed: \(msg)"
        case .invalidInput(let msg): return "Invalid input: \(msg)"
        }
    }
}

public final class EmbeddingModel {
    private let model: MLModel
    private let modelConfig: MLModelConfiguration
    public let embeddingDim: Int
    public let supportedSeqLengths: [Int]

    public init(modelPath: String, computeUnits: MLComputeUnits, embeddingDim: Int = 768) throws {
        let url = URL(fileURLWithPath: modelPath)

        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw EmbeddingModelError.modelNotFound(modelPath)
        }

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        do {
            // An .mlpackage must be compiled to .mlmodelc before it can be loaded.
            if url.pathExtension == "mlpackage" {
                let compiledURL = try MLModel.compileModel(at: url)
                self.model = try MLModel(contentsOf: compiledURL, configuration: config)
            } else {
                self.model = try MLModel(contentsOf: url, configuration: config)
            }
        } catch {
            throw EmbeddingModelError.compilationFailed(error.localizedDescription)
        }

        self.modelConfig = config
        self.embeddingDim = embeddingDim
        self.supportedSeqLengths = Self.detectSupportedLengths(model: model)

        print("[EmbeddingModel] Loaded: \(url.lastPathComponent)")
        print("[EmbeddingModel] Compute units: \(computeUnits.rawValue)")
        print("[EmbeddingModel] Supported seq lengths: \(supportedSeqLengths)")
        print("[EmbeddingModel] Embedding dim: \(embeddingDim)")
    }

    public func predict(inputIds: [Int], attentionMask: [Int], seqLength: Int) throws -> [Float16] {
        guard inputIds.count == seqLength else {
            throw EmbeddingModelError.invalidInput(
                "inputIds length \(inputIds.count) != seqLength \(seqLength)"
            )
        }

        guard attentionMask.count == seqLength else {
            throw EmbeddingModelError.invalidInput(
                "attentionMask length \(attentionMask.count) != seqLength \(seqLength)"
            )
        }

        let inputIdsArray = try MLMultiArray(shape: [1, NSNumber(value: seqLength)], dataType: .int32)
        let ip = inputIdsArray.dataPointer.bindMemory(to: Int32.self, capacity: seqLength)
        for i in 0..<seqLength { ip[i] = Int32(inputIds[i]) }

        // Model input `attention_mask` is declared fp16 (1.0 valid, 0.0 pad).
        // Write raw fp16 bits: 0x3C00 == 1.0 in fp16.
        let attentionMaskArray = try MLMultiArray(shape: [1, NSNumber(value: seqLength)], dataType: .float16)
        let ap = attentionMaskArray.dataPointer.bindMemory(to: UInt16.self, capacity: seqLength)
        let fp16One: UInt16 = 0x3C00
        for i in 0..<seqLength { ap[i] = attentionMask[i] == 0 ? 0 : fp16One }

        let input = try MLDictionaryFeatureProvider(
            dictionary: [
                "input_ids": MLFeatureValue(multiArray: inputIdsArray),
                "attention_mask": MLFeatureValue(multiArray: attentionMaskArray),
            ]
        )

        let output = try model.prediction(from: input)

        guard let embeddingValue = output.featureValue(for: "embedding"),
              let embeddingArray = embeddingValue.multiArrayValue
        else {
            throw EmbeddingModelError.predictionFailed("Missing 'embedding' output")
        }

        // Output is (1, 768) fp16. Read directly as Float16 — no conversion.
        let count = embeddingArray.count
        let src = embeddingArray.dataPointer.bindMemory(to: Float16.self, capacity: count)
        return Array(UnsafeBufferPointer(start: src, count: count))
    }

    public func selectBucket(for tokenCount: Int) -> Int {
        for length in supportedSeqLengths.sorted() {
            if tokenCount <= length { return length }
        }
        return supportedSeqLengths.max() ?? 2048
    }

    private static func detectSupportedLengths(model: MLModel) -> [Int] {
        let inputDescription = model.modelDescription.inputDescriptionsByName
        var lengths: [Int] = []

        if let inputIdsDesc = inputDescription["input_ids"],
           let multiArrayConstraint = inputIdsDesc.multiArrayConstraint {
            let shape = multiArrayConstraint.shape.map { $0.intValue }
            if shape.count == 2, shape[1] > 0 {
                lengths.append(shape[1])
            }
        }

        if lengths.isEmpty {
            lengths = [128, 256, 512, 1024, 2048]
        }

        return lengths.sorted()
    }
}
