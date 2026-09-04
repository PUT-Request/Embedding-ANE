// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "EmbeddingANE",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "EmbeddingServer", targets: ["EmbeddingServer"]),
        .library(name: "EmbeddingPipeline", targets: ["EmbeddingPipeline"]),
    ],
    dependencies: [
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0"),
        .package(url: "https://github.com/apple/swift-nio.git", from: "2.65.0"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "0.1.8"),
    ],
    targets: [
        .target(
            name: "EmbeddingPipeline",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "EmbeddingServer",
            dependencies: [
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "HummingbirdCore", package: "hummingbird"),
                .product(name: "NIOCore", package: "swift-nio"),
                .target(name: "EmbeddingPipeline"),
            ]
        ),
    ]
)
