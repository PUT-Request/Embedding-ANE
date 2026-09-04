import Foundation
import CoreML

struct ServerConfig {
    let host: String
    let port: Int
    let modelPath: String
    let modelName: String
    let maxBatchSize: Int
    let defaultDimensions: Int
    let computeUnits: String

    static func fromArguments() -> ServerConfig {
        let args = CommandLine.arguments

        func value(for flag: String, default defaultValue: String) -> String {
            guard let index = args.firstIndex(of: flag),
                  index + 1 < args.count
            else { return defaultValue }
            return args[index + 1]
        }

        return ServerConfig(
            host: value(for: "--host", default: "127.0.0.1"),
            port: Int(value(for: "--port", default: "6333")) ?? 6333,
            modelPath: value(for: "--model", default: "./Models/embeddinggemma-300m-coreml/encoder.mlmodelc"),
            modelName: value(for: "--model-name", default: "embeddinggemma-300m"),
            maxBatchSize: Int(value(for: "--max-batch", default: "32")) ?? 32,
            defaultDimensions: Int(value(for: "--dimensions", default: "768")) ?? 768,
            computeUnits: value(for: "--compute-units", default: "cpuAndNeuralEngine")
        )
    }

    var computeUnitConfig: MLComputeUnits {
        switch computeUnits {
        case "cpuOnly": return .cpuOnly
        case "cpuAndGPU": return .cpuAndGPU
        case "all": return .all
        default: return .cpuAndNeuralEngine
        }
    }
}
