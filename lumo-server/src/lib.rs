pub mod auth;
pub mod config;
use actix_web::{dev::Server, get, post, web::Json, App, HttpResponse, HttpServer, Responder};
use anyhow::Result;
use base64::{self, Engine};
use std::pin::Pin;
use config::Servers;
use lumo::{
    agent::{Agent, FunctionCallingAgentBuilder, AgentStream},
    models::{openai::{OpenAIServerModelBuilder, Status}, types::Message},
    tools::{
        exa_search::ExaSearchTool, AsyncTool, DuckDuckGoSearchTool, GoogleSearchTool,
        VisitWebsiteTool,
    },
};
#[cfg(feature = "code")]
use lumo::agent::CodeAgentBuilder;
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
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_sdk::trace::{self as sdktrace, BatchConfigBuilder, BatchSpanProcessor};
use tracing::instrument;
use tokio::sync::broadcast;
use actix_web::web::Bytes;
use futures::StreamExt;

#[cfg(feature = "code")]
use lumo::tools::PythonInterpreterTool;
#[cfg(feature = "mcp")]
use {
    lumo::agent::McpAgentBuilder,

};

use actix_cors::Cors;
use actix_web::http::header;
use dotenv::dotenv;
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
    #[serde(skip_serializing_if = "Option::is_none")]
    max_results: Option<usize>,
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
    ExaSearchTool,
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
            "ExaSearchTool" => Ok(ToolType::ExaSearchTool),
            #[cfg(feature = "code")]
            "PythonInterpreter" => Ok(ToolType::PythonInterpreter),
            _ => Err(actix_web::error::ErrorBadRequest(format!(
                "Invalid tool type: {}",
                s
            ))),
        }
    }
}

fn create_tool(tool_type: &ToolType, max_results: Option<usize>) -> Box<dyn AsyncTool> {
    match tool_type {
        ToolType::DuckDuckGo => Box::new(DuckDuckGoSearchTool::new()),
        ToolType::VisitWebsite => Box::new(VisitWebsiteTool::new()),
        ToolType::GoogleSearchTool => Box::new(GoogleSearchTool::new(None)),
        ToolType::ExaSearchTool => Box::new(ExaSearchTool::new(max_results.unwrap_or(5), None)),
        #[cfg(feature = "code")]
        ToolType::PythonInterpreter => Box::new(PythonInterpreterTool::new()),
    }
}

pub fn init_tracer() -> Option<SdkTracerProvider> {
    dotenv().ok();

    let (langfuse_public_key, langfuse_secret_key, endpoint) = if cfg!(debug_assertions) {
        match (
            std::env::var("LANGFUSE_PUBLIC_KEY_DEV"),
            std::env::var("LANGFUSE_SECRET_KEY_DEV"),
            std::env::var("LANGFUSE_HOST_DEV"),
        ) {
            (Ok(public_key), Ok(secret_key), Ok(host)) => (
                public_key,
                secret_key,
                format!("{}/api/public/otel/v1/traces", host),
            ),
            _ => return None, // If any key is missing, return None to disable tracing
        }
    } else {
        match (
            std::env::var("LANGFUSE_PUBLIC_KEY"),
            std::env::var("LANGFUSE_SECRET_KEY"),
            std::env::var("LANGFUSE_HOST"),
        ) {
            (Ok(public_key), Ok(secret_key), Ok(host)) => (
                public_key,
                secret_key,
                format!("{}/api/public/otel/v1/traces", host),
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

    let otlp_exporter = match opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_endpoint(endpoint)
        .with_protocol(opentelemetry_otlp::Protocol::HttpBinary)
        .with_headers(headers)
        .build()
    {
        Ok(exporter) => exporter,
        Err(_) => return None,
    };

    let batch = BatchSpanProcessor::builder(otlp_exporter)
        .with_batch_config(
            BatchConfigBuilder::default()
                .with_max_queue_size(512)
                .build(),
        )
        .build();

    let tracer_provider = sdktrace::SdkTracerProvider::builder()
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
                            std::env::var("ENVIRONMENT")
                                .unwrap_or_else(|_| "production".to_string())
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

    Some(tracer_provider)
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
        .with_start_time(std::time::SystemTime::now())
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
            use rmcp::{transport::{ConfigureCommandExt, TokioChildProcess}, ServiceExt};
            use tokio::process::Command;
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

                let client = ().serve(TokioChildProcess::new(Command::new(&server_config.command).configure(|cmd| {
                    cmd.args(&server_config.args);
                })).map_err(actix_web::error::ErrorInternalServerError)?)
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
                .with_context(cx.clone())
                .await
                .map_err(actix_web::error::ErrorInternalServerError)?
        }

        #[cfg(feature = "code")]
        Some("code-agent") => {
            let tools = if let Some(tools) = &req.tools {
                tools
                    .iter()
                    .map(|tool| ToolType::from_str(tool).map(|t| create_tool(&t, req.max_results)))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                vec![]
            };
            let mut agent = CodeAgentBuilder::new(model)
                .with_tools(tools)
                .with_max_steps(req.max_steps)
                .with_history(req.history.clone())
                .with_logging_level(Some(log::LevelFilter::Info))
                .build()
                .map_err(actix_web::error::ErrorInternalServerError)?;

            agent
                .run(&req.task, false)
                .with_context(cx.clone())
                .await
                .map_err(actix_web::error::ErrorInternalServerError)?
        }
        _ => {
            // Default function calling agent logic...
            let servers = Servers::load().map_err(actix_web::error::ErrorInternalServerError)?;

            let tools = if let Some(tools) = &req.tools {
                tools
                    .iter()
                    .map(|tool| ToolType::from_str(tool).map(|t| create_tool(&t, req.max_results)))
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
    cx.span().end_with_timestamp(std::time::SystemTime::now());

    Ok(Json(RunTaskResponse { response }))
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum StreamEvent {
    #[serde(rename = "token")]
    Token { content: String },
    #[serde(rename = "step")]
    Step { step: serde_json::Value },
    #[serde(rename = "error")]
    Error { message: String },
    #[serde(rename = "done")]
    Done,
}

#[post("/stream")]
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
async fn stream_task(req: Json<RunTaskRequest>) -> Result<HttpResponse, actix_web::Error> {
    let tracer = global::tracer("lumo");
    let span = tracer
        .span_builder("stream_task")
        .with_kind(SpanKind::Server)
        .with_start_time(std::time::SystemTime::now())
        .with_attributes(vec![
            KeyValue::new("gen_ai.operation.name", "stream_task"),
            KeyValue::new("gen_ai.task", req.task.clone()),
            KeyValue::new("gen_ai.base_url", req.base_url.clone()),
            KeyValue::new("input.value", req.task.clone()),
            KeyValue::new("timestamp", chrono::Utc::now().to_rfc3339()),
        ])
        .start(&tracer);
    let cx = Context::current_with_span(span);

    // Get API key based on base URL
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

    // Create broadcast channel for token-level streaming
    let (tx, rx) = broadcast::channel::<Status>(2000);
    let task_str = req.task.clone();

    // Create SSE stream - construct the entire stream inside async_stream to own the agent
    let sse_stream = match req.agent_type.as_deref() {
        #[cfg(feature = "mcp")]
        Some("mcp") => {
            use lumo::agent::McpAgentBuilder;

            use rmcp::{transport::{ConfigureCommandExt, TokioChildProcess}, ServiceExt};
            use tokio::process::Command;
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

                let client = ().serve(TokioChildProcess::new(Command::new(&server_config.command).configure(|cmd| {
                    cmd.args(&server_config.args);
                })).map_err(actix_web::error::ErrorInternalServerError)?)
                .await
                .map_err(actix_web::error::ErrorInternalServerError)?;

                clients.push(client);
            }

            // Create and run MCP agent with filtered clients
            let agent = McpAgentBuilder::new(model)
                .with_system_prompt(servers.system_prompt.as_deref())
                .with_max_steps(req.max_steps)
                .with_history(req.history.clone())
                .with_mcp_clients(clients)
                .with_logging_level(Some(log::LevelFilter::Info))
                .build()
                .await
                .map_err(actix_web::error::ErrorInternalServerError)?;

            create_agent_stream(agent, task_str, tx, rx, cx)
        }

        #[cfg(feature = "code")]
        Some("code-agent") => {
            let tools = if let Some(tools) = &req.tools {
                tools
                    .iter()
                    .map(|tool| ToolType::from_str(tool).map(|t| create_tool(&t, req.max_results)))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                vec![]
            };
            let agent = CodeAgentBuilder::new(model)
                .with_tools(tools)
                .with_max_steps(req.max_steps)
                .with_history(req.history.clone())
                .with_logging_level(Some(log::LevelFilter::Info))
                .build()
                .map_err(actix_web::error::ErrorInternalServerError)?;

            create_agent_stream(agent, task_str, tx, rx, cx)
        }
        _ => {
            // Default function calling agent logic
            let servers = Servers::load().map_err(actix_web::error::ErrorInternalServerError)?;

            let tools = if let Some(tools) = &req.tools {
                tools
                    .iter()
                    .map(|tool| ToolType::from_str(tool).map(|t| create_tool(&t, req.max_results)))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                vec![]
            };

            let agent = FunctionCallingAgentBuilder::new(model)
                .with_tools(tools)
                .with_max_steps(req.max_steps)
                .with_history(req.history.clone())
                .with_system_prompt(servers.system_prompt.as_deref())
                .with_logging_level(Some(log::LevelFilter::Info))
                .build()
                .map_err(actix_web::error::ErrorInternalServerError)?;

            create_agent_stream(agent, task_str, tx, rx, cx)
        }
    };

    Ok(HttpResponse::Ok()
        .content_type("text/event-stream")
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("Connection", "keep-alive"))
        .insert_header(("X-Accel-Buffering", "no"))
        .streaming(sse_stream))
}

fn create_agent_stream<A>(
    mut agent: A,
    task: String,
    tx: broadcast::Sender<Status>,
    mut rx: broadcast::Receiver<Status>,
    cx: Context,
) -> Pin<Box<dyn futures::Stream<Item = Result<Bytes, std::io::Error>>>>
where
    A: AgentStream + 'static,
{
    Box::pin(
    async_stream::stream! {
        // Get the stream from the agent
        let stream = match agent.stream_run(&task, false, Some(tx)) {
            Ok(s) => s,
            Err(e) => {
                let event = StreamEvent::Error { 
                    message: e.to_string() 
                };
                if let Ok(json) = serde_json::to_string(&event) {
                    yield Ok(Bytes::from(format!("data: {}\n\n", json)));
                }
                return;
            }
        };

        // Pin the stream for iteration
        tokio::pin!(stream);

        // Use select to poll both the step stream and token receiver simultaneously
        loop {
            tokio::select! {
                // Poll for tokens continuously
                status = rx.recv() => {
                    match status {
                        Ok(Status::FirstContent(content)) | Ok(Status::Content(content)) => {
                            let event = StreamEvent::Token { content };
                            if let Ok(json) = serde_json::to_string(&event) {
                                yield Ok(Bytes::from(format!("data: {}\n\n", json)));
                            }
                        }
                        Ok(Status::ToolCallStart(tool_name)) => {
                            let event = StreamEvent::Token { 
                                content: format!("[Using tool: {}]", tool_name) 
                            };
                            if let Ok(json) = serde_json::to_string(&event) {
                                yield Ok(Bytes::from(format!("data: {}\n\n", json)));
                            }
                        }
                        Err(broadcast::error::RecvError::Lagged(skipped)) => {
                            // Log that we skipped some messages but continue
                            log::warn!("Skipped {} messages due to lag", skipped);
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            // Channel closed, break to drain steps
                            break;
                        }
                        _ => {}
                    }
                }
                // Poll for steps
                step_result = stream.next() => {
                    match step_result {
                        Some(Ok(step)) => {
                            // Send the step event
                            let step_value = serde_json::to_value(&step).unwrap_or_default();
                            let event = StreamEvent::Step { step: step_value };
                            if let Ok(json) = serde_json::to_string(&event) {
                                yield Ok(Bytes::from(format!("data: {}\n\n", json)));
                            }
                        }
                        Some(Err(e)) => {
                            let event = StreamEvent::Error { 
                                message: e.to_string() 
                            };
                            if let Ok(json) = serde_json::to_string(&event) {
                                yield Ok(Bytes::from(format!("data: {}\n\n", json)));
                            }
                            break;
                        }
                        None => {
                            // Step stream ended
                            break;
                        }
                    }
                }
            }
        }

        // Drain any remaining tokens after steps complete
        while let Ok(status) = rx.try_recv() {
            match status {
                Status::FirstContent(content) | Status::Content(content) => {
                    let event = StreamEvent::Token { content };
                    if let Ok(json) = serde_json::to_string(&event) {
                        yield Ok(Bytes::from(format!("data: {}\n\n", json)));
                    }
                }
                _ => {}
            }
        }

        // Send done event
        let event = StreamEvent::Done;
        if let Ok(json) = serde_json::to_string(&event) {
            yield Ok(Bytes::from(format!("data: {}\n\n", json)));
        }

        cx.span().end_with_timestamp(std::time::SystemTime::now());
    })
}

pub fn run(listener: TcpListener) -> std::io::Result<Server> {
    Ok(HttpServer::new(move || {
        println!("Config File Path: {:?}", Servers::config_path().unwrap());
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
            .service(stream_task)
    })
    .listen(listener)?
    .run())
}
