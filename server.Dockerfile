FROM lukemathwalker/cargo-chef:latest AS chef
WORKDIR /app
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev 


FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# Build dependencies - this is the caching Docker layer!
RUN cargo chef cook --release --recipe-path recipe.json
# Build application
COPY . .
RUN cargo build --release --package lumo-server --no-default-features

# We do not need the Rust toolchain to run the binary!
FROM debian:bookworm-slim AS runtime
WORKDIR /app

# Combine RUN commands and cleanup in the same layer
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    openssl \
    libcurl4-openssl-dev

COPY --from=builder /app/target/release/lumo-server /usr/local/bin
COPY --from=builder /app/lumo-server/src/config/servers.yaml /usr/local/bin/src/config/servers.yaml

EXPOSE 8080

# Use ARG for build-time variables (these won't be stored in the image)
ARG OPENAI_API_KEY
ARG GOOGLE_API_KEY
ARG GROQ_API_KEY
ARG ANTHROPIC_API_KEY
ARG EXA_API_KEY
ARG LUMO_API_KEY
ARG ENABLE_AUTH

# Set runtime environment variables from ARGs
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV GOOGLE_API_KEY=${GOOGLE_API_KEY}
ENV GROQ_API_KEY=${GROQ_API_KEY}
ENV ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
ENV EXA_API_KEY=${EXA_API_KEY}
ENV LUMO_API_KEY=${LUMO_API_KEY}
ENV ENABLE_AUTH=${ENABLE_AUTH}

ENTRYPOINT ["/usr/local/bin/lumo-server"] 