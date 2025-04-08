use opentelemetry::KeyValue;
use opentelemetry_otlp::{WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::trace::{BatchConfigBuilder, BatchSpanProcessor, SdkTracerProvider, TraceError};
use opentelemetry::trace::TracerProvider;
use base64::Engine;

pub fn init_tracer() -> Result<SdkTracerProvider, TraceError> {
    let (langfuse_public_key, langfuse_secret_key, endpoint) = if cfg!(debug_assertions) {
        (
            "pk-lf-bc752d0c-f1f1-416d-8792-5ee6f580137e".to_string(),
            "sk-lf-814738c0-c864-44b4-9ed5-9dcf4c258d2d".to_string(),
            "http://localhost:3000/api/public/otel/v1/traces".to_string(),
        )
    } else {
        (
            std::env::var("LANGFUSE_PUBLIC_KEY").expect("LANGFUSE_PUBLIC_KEY must be set"),
            std::env::var("LANGFUSE_SECRET_KEY").expect("LANGFUSE_SECRET_KEY must be set"),
            format!("{}/api/public/otel/v1/traces", std::env::var("LANGFUSE_HOST").expect("LANGFUSE_HOST must be set")),
        )
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
        .with_endpoint(endpoint)
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

    Ok(tracer_provider)
}
