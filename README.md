# EmbeddingANE

Run [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) on the **Apple Neural Engine** via CoreML. OpenAI-compatible local HTTP server built with Swift Package Manager.

## Requirements

- macOS 14+ on Apple Silicon (M1–M4)
- Xcode Command Line Tools (`xcode-select --install`)
- Python 3 + `pip install huggingface_hub`

## Setup

```bash
# Download the ANE-optimized CoreML model
./Scripts/download-model.sh

# Build and resolve dependencies
./Scripts/setup.sh

# Run the server
./Scripts/run.sh
```

Server starts at `http://127.0.0.1:6333`.

## Usage

```bash
curl http://localhost:6333/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "embeddinggemma-300m", "input": "What is the Red Planet?"}'
```

### `GET /v1/models` — List models
### `GET /health` — Health check

| Parameter     | Type               | Description                                  |
|---------------|--------------------|----------------------------------------------|
| `model`       | string             | `embeddinggemma-300m`                         |
| `input`       | string \| string[] | Single text or batch                          |
| `dimensions`  | int (optional)     | Truncate to 128, 256, 512, or 768 (default)  |

## Configuration

```bash
swift run EmbeddingServer --port 9000 --dimensions 256
```

| Flag              | Default     | Description               |
|-------------------|-------------|---------------------------|
| `--host`          | 127.0.0.1   | Bind address              |
| `--port`          | 6333        | Listen port               |
| `--max-batch`     | 32          | Max texts per request     |
| `--dimensions`    | 768         | Default embedding dim     |
| `--compute-units` | cpuAndNeuralEngine | CoreML compute units |

## Project Structure

```
embedding-ane/
├── Package.swift
├── Sources/
│   ├── EmbeddingServer/       # HTTP server + OpenAI-compatible routes
│   └── EmbeddingPipeline/     # CoreML inference wrapper
├── Scripts/
│   ├── download-model.sh      # Fetch CoreML model from HuggingFace
│   ├── setup.sh               # Build
│   ├── run.sh                 # Launch server
│   └── conversion/            # Optional: local HF → CoreML converter
└── Models/                    # CoreML model bundle (after download)
```

## License

Model weights: Google Gemma license. Code: see repository.
