use std::net::TcpListener;

use lumo_server::{init_tracer, run};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, fmt};
use tracing_opentelemetry;

#[actix_web::main]
#[tracing::instrument]
async fn main() -> std::io::Result<()> {
    let _provider = init_tracer().await.expect("Failed to initialize OpenTelemetry tracer");

    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(fmt::layer())
        .with(tracing_opentelemetry::layer())
        .init();

    let listener = TcpListener::bind("0.0.0.0:8080")?;
    println!("Listening on 0.0.0.0:8080");
    run(listener).expect("Failed to bind address").await
}
