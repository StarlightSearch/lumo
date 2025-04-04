pub mod auth;
pub mod config;
use actix_web::{dev::Server, get, post, web::Json, App, HttpResponse, HttpServer, Responder};
use anyhow::Result;
use base64::{self, Engine};
use config::Servers;
use lumo::{
    agent::{Agent, FunctionCallingAgentBuilder},
    models::{openai::OpenAIServerModelBuilder, types::Message},
    tools::{AsyncTool, DuckDuckGoSearchTool, GoogleSearchTool, VisitWebsiteTool},
};
use opentelemetry::trace::FutureExt;
use opentelemetry::trace::Tracer;
use opentelemetry::trace::TracerProvider;
use opentelemetry::Context;
use opentelemetry::KeyValue;
use opentelemetry::{
    global,
    trace::{SpanKind, TraceContextExt},
};
use opentelemetry_otlp::{WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::trace::{self as sdktrace, BatchConfigBuilder, BatchSpanProcessor};
use opentelemetry_sdk::trace::{SdkTracerProvider, TraceError};
use tracing::instrument;

#[cfg(feature = "code")]
use lumo::tools::PythonInterpreterTool;
#[cfg(feature = "mcp")]
use {
    lumo::agent::McpAgentBuilder,
    mcp_client::{
        ClientCapabilities, ClientInfo, McpClient, McpClientTrait, McpService, StdioTransport,
        Transport,
    },
};

use actix_cors::Cors;
use actix_web::http::header;
use serde::{Deserialize, Serialize};
use std::net::TcpListener;
use std::str::FromStr;

#[derive(Deserialize)]
struct RunTaskRequest {
    task: String,
    model: String,
    base_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_steps: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    history: Option<Vec<Message>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_type: Option<String>,
}

#[derive(Serialize)]
struct RunTaskResponse {
    response: String,
}

#[derive(Debug, Clone, Deserialize)]
enum ToolType {
    DuckDuckGo,
    VisitWebsite,
    GoogleSearchTool,
    #[cfg(feature = "code")]
    PythonInterpreter,
}

impl FromStr for ToolType {
    type Err = actix_web::error::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "DuckDuckGo" => Ok(ToolType::DuckDuckGo),
            "VisitWebsite" => Ok(ToolType::VisitWebsite),
            "GoogleSearchTool" => Ok(ToolType::GoogleSearchTool),
            #[cfg(feature = "code")]
            "PythonInterpreter" => Ok(ToolType::PythonInterpreter),
            _ => Err(actix_web::error::ErrorBadRequest(format!(
                "Invalid tool type: {}",
                s
            ))),
        }
    }
}

fn create_tool(tool_type: &ToolType) -> Box<dyn AsyncTool> {
    match tool_type {
        ToolType::DuckDuckGo => Box::new(DuckDuckGoSearchTool::new()),
        ToolType::VisitWebsite => Box::new(VisitWebsiteTool::new()),
        ToolType::GoogleSearchTool => Box::new(GoogleSearchTool::new(None)),
        #[cfg(feature = "code")]
        ToolType::PythonInterpreter => Box::new(PythonInterpreterTool::new()),
    }
}

pub fn init_tracer() -> Result<SdkTracerProvider, TraceError> {
    let (langfuse_public_key, langfuse_secret_key, endpoint) = if cfg!(debug_assertions) {
        (
            "pk-lf-b3403582-e669-4ebf-91df-b80921f13b12".to_string(),
            "sk-lf-01e1ea91-570a-44a6-9543-dc803a931041".to_string(),
            "http://localhost:3000/api/public/otel/v1/traces".to_string(),
        )
    } else {
        (
            std::env::var("LANGFUSE_PUBLIC_KEY").expect("LANGFUSE_PUBLIC_KEY must be set"),
            std::env::var("LANGFUSE_SECRET_KEY").expect("LANGFUSE_SECRET_KEY must be set"),
            std::env::var("LANGFUSE_HOST").expect("LANGFUSE_HOST must be set") + "/api/public/otel/v1/traces",
        )
    };

    // Basic Auth: base64(public_key:secret_key)
    let auth_header = format!(
        "Basic {}",
        base64::engine::general_purpose::STANDARD
            .encode(format!("{}:{}", langfuse_public_key, langfuse_secret_key))
    );
    println!("auth_header: {}", auth_header);

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
                .with_scheduled_delay(std::time::Duration::from_millis(100))
                .build(),
        )
        .build();

    let tracer_provider = sdktrace::SdkTracerProvider::builder()
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

#[get("/health_check")]
#[instrument]
async fn health_check() -> impl Responder {
    HttpResponse::Ok()
}

#[post("/run")]
#[instrument(
    skip(req),
    fields(
        task = %req.task,
        model = %req.model,
        base_url = %req.base_url,
        tools = ?req.tools,
        max_steps = ?req.max_steps,
        agent_type = ?req.agent_type
    )
)]

async fn run_task(req: Json<RunTaskRequest>) -> Result<impl Responder, actix_web::Error> {
    let tracer = global::tracer("lumo");
    let span = tracer
        .span_builder("run_task")
        .with_kind(SpanKind::Server)
        .with_attributes(vec![
            KeyValue::new("gen_ai.operation.name", "run_task"),
            KeyValue::new("gen_ai.task", req.task.clone()),
            KeyValue::new("gen_ai.base_url", req.base_url.clone()),
            KeyValue::new("input.value", req.task.clone()),
            KeyValue::new("timestamp", chrono::Utc::now().to_rfc3339()),
        ])
        .start(&tracer);
    let cx = Context::current_with_span(span);
    // use base url to get the right key from environment variables
    let api_key = if req.base_url == "https://api.openai.com/v1/chat/completions" {
        std::env::var("OPENAI_API_KEY").ok()
    } else if req.base_url
        == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    {
        std::env::var("GOOGLE_API_KEY").ok()
    } else if req.base_url.to_lowercase().contains("groq") {
        std::env::var("GROQ_API_KEY").ok()
    } else if req.base_url.to_lowercase().contains("anthropic") {
        std::env::var("ANTHROPIC_API_KEY").ok()
    } else {
        None
    };

    cx.span()
        .set_attribute(KeyValue::new("gen_ai.system", req.base_url.clone()));

    let model = OpenAIServerModelBuilder::new(&req.model)
        .with_base_url(Some(&req.base_url))
        .with_api_key(api_key.as_deref())
        .build()
        .map_err(actix_web::error::ErrorInternalServerError)?;

    let response = match req.agent_type.as_deref() {
        #[cfg(feature = "mcp")]
        Some("mcp") => {
            // Create fresh clients for this request
            let mut clients = Vec::new();
            let servers = Servers::load().map_err(actix_web::error::ErrorInternalServerError)?;

            // Only create clients for requested tools
            for (server_name, server_config) in servers.servers.iter() {
                // Skip this server if its tools aren't requested
                if let Some(tools) = &req.tools {
                    if !tools.contains(&server_name.to_string()) {
                        continue;
                    }
                }

                let transport = StdioTransport::new(
                    &server_config.command,
                    server_config.args.clone(),
                    server_config.env.clone().unwrap_or_default(),
                );

                let transport_handle = transport
                    .start()
                    .await
                    .map_err(actix_web::error::ErrorInternalServerError)?;

                let service = McpService::new(transport_handle);
                let mut client = McpClient::new(service);

                client
                    .initialize(
                        ClientInfo {
                            name: format!("{}-client", server_name),
                            version: "1.0.0".into(),
                        },
                        ClientCapabilities::default(),
                    )
                    .await
                    .map_err(actix_web::error::ErrorInternalServerError)?;
                clients.push(client);
            }

            // Create and run MCP agent with filtered clients
            let mut agent = McpAgentBuilder::new(model)
                .with_system_prompt(servers.system_prompt.as_deref())
                .with_max_steps(req.max_steps)
                .with_history(req.history.clone())
                .with_mcp_clients(clients)
                .with_logging_level(Some(log::LevelFilter::Info))
                .build()
                .await
                .map_err(actix_web::error::ErrorInternalServerError)?;

            agent
                .run(&req.task, false)
                .with_context(Context::current_with_span(span))
                .await
                .map_err(actix_web::error::ErrorInternalServerError)?
        }
        _ => {
            // Default function calling agent logic...
            let servers = Servers::load().map_err(actix_web::error::ErrorInternalServerError)?;

            let tools = if let Some(tools) = &req.tools {
                tools
                    .iter()
                    .map(|tool| ToolType::from_str(tool).map(|t| create_tool(&t)))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                vec![]
            };

            let mut agent = FunctionCallingAgentBuilder::new(model)
                .with_tools(tools)
                .with_max_steps(req.max_steps)
                .with_history(req.history.clone())
                .with_system_prompt(servers.system_prompt.as_deref())
                .with_logging_level(Some(log::LevelFilter::Info))
                .build()
                .map_err(actix_web::error::ErrorInternalServerError)?;

            agent
                .run(&req.task, false)
                .with_context(cx.clone())
                .await
                .map_err(actix_web::error::ErrorInternalServerError)?
        }
    };
    cx.span()
        .set_attribute(KeyValue::new("output.value", response.clone()));

    Ok(Json(RunTaskResponse { response }))
}

pub fn run(listener: TcpListener) -> std::io::Result<Server> {
    Ok(HttpServer::new(move || {
        let _ = Servers::load().map_err(actix_web::error::ErrorInternalServerError);
        let cors = Cors::default()
            .allow_any_origin()
            .allowed_methods(vec!["GET", "POST"])
            .allowed_headers(vec![
                header::AUTHORIZATION,
                header::ACCEPT,
                header::CONTENT_TYPE,
            ])
            .max_age(3600);

        App::new()
            .wrap(cors)
            .wrap(auth::ApiKeyAuth)
            .service(health_check)
            .service(run_task)
    })
    .listen(listener)?
    .run())
}
