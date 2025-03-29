# Rust builder with cargo-chef
FROM lukemathwalker/cargo-chef:latest AS chef
WORKDIR /app
RUN apt-get update && apt-get install -y python3.11-dev

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# Build dependencies - this layer is cached as long as dependencies don't change
RUN cargo chef cook --release --recipe-path recipe.json
# Build application
COPY . .
# Build with minimal features and optimize for size
RUN cargo build --release \
    && strip /app/target/release/lumo

# Use distroless as runtime image
FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y python3.11-dev

WORKDIR /app
# Copy only the binary
COPY --from=builder /app/target/release/lumo /usr/local/bin/
# Create config directory
WORKDIR /root/.config/lumo

# Set the entrypoint to the binary
ENTRYPOINT ["/usr/local/bin/lumo"]
# Set default arguments that can be overridden
CMD ["--model-type", "ollama", "--model-id", "qwen2.5", "--tools", "duck-duck-go,visit-website"]
