import Hummingbird

@main
struct EmbeddingServer {
    static func main() async throws {
        let config = ServerConfig.fromArguments()

        print("""
        ╔══════════════════════════════════════════════════════╗
        ║          EmbeddingGemma-300M Local Server            ║
        ║          Powered by Apple Neural Engine             ║
        ╚══════════════════════════════════════════════════════╝
        """)

        let app = try await buildApp(config)

        print("[Server] Listening on http://\(config.host):\(config.port)")
        print("[Server] API: POST /v1/embeddings")
        print("[Server] Models: GET /v1/models")
        print("[Server] Health: GET /health")
        print("")

        try await app.runService()
    }
}
