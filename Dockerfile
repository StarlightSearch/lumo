# Rust builder with cargo-chef
FROM lukemathwalker/cargo-chef:latest AS chef
WORKDIR /app

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
FROM gcr.io/distroless/cc-debian12 AS runtime
WORKDIR /app
# Copy only the binary
COPY --from=builder /app/target/release/lumo /usr/local/bin/
# Create config directory
WORKDIR /root/.config/lumo

ENTRYPOINT ["/usr/local/bin/lumo"]
