use opentelemetry::KeyValue;
use opentelemetry_otlp::{WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::trace::{BatchConfigBuilder, BatchSpanProcessor, SdkTracerProvider};
use opentelemetry::trace::TracerProvider;
use base64::Engine;
use std::env;
use dotenv::dotenv;

pub fn init_tracer() -> Option<(SdkTracerProvider, String)> {
    dotenv().ok();

    let (langfuse_public_key, langfuse_secret_key, endpoint, host) = if cfg!(debug_assertions) {
        match (env::var("LANGFUSE_PUBLIC_KEY_DEV"), env::var("LANGFUSE_SECRET_KEY_DEV"), env::var("LANGFUSE_HOST_DEV")) {
            (Ok(public_key), Ok(secret_key), Ok(host)) => (
                public_key,
                secret_key,
                format!("{}/api/public/otel/v1/traces", host),
                host,
            ),
            _ => return None, // If any key is missing, return None to disable tracing
        }
    } else {
        match (env::var("LANGFUSE_PUBLIC_KEY"), env::var("LANGFUSE_SECRET_KEY"), env::var("LANGFUSE_HOST")) {
            (Ok(public_key), Ok(secret_key), Ok(host)) => (
                public_key,
                secret_key,
                format!("{}/api/public/otel/v1/traces", host),
                host,
            ),
            _ => return None, // If any key is missing, return None to disable tracing
        }
    };

    // Basic Auth: base64(public_key:secret_key)
    let auth_header = format!(
        "Basic {}",
        base64::engine::general_purpose::STANDARD
            .encode(format!("{}:{}", langfuse_public_key, langfuse_secret_key))
    );

    let mut headers = std::collections::HashMap::new();
    headers.insert("Authorization".to_string(), auth_header);

    let otlp_exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_endpoint(endpoint.clone())
        .with_protocol(opentelemetry_otlp::Protocol::HttpBinary)
        .with_headers(headers)
        // .build()
        // .with_tonic().with_endpoint("http://localhost:4317")
        .build()
        .unwrap();

    let batch = BatchSpanProcessor::builder(otlp_exporter)
        .with_batch_config(
            BatchConfigBuilder::default()
                .with_max_queue_size(512)
                .build(),
        )
        .build();

    let tracer_provider = SdkTracerProvider::builder()
        // .with_span_processor(batch_processor)
        .with_span_processor(batch)
        .with_resource(
            opentelemetry_sdk::resource::Resource::builder()
                .with_service_name("lumo")
                .with_attributes(vec![
                    KeyValue::new(
                        "deployment.environment",
                        if cfg!(debug_assertions) {
                            "development".to_string()
                        } else {
                            std::env::var("ENVIRONMENT").unwrap_or_else(|_| "production".to_string())
                        },
                    ),
                    KeyValue::new("deployment.name", "lumo"),
                    KeyValue::new("deployment.version", env!("CARGO_PKG_VERSION")),
                ])
                .build(),
        )
        .build();

    // Initialize the tracer
    let _ = tracer_provider.tracer("lumo");

    // Set the global tracer provider
    opentelemetry::global::set_tracer_provider(tracer_provider.clone());

    Some((tracer_provider, host))
}
