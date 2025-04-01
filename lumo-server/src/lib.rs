pub mod config;
pub mod auth;
use actix_web::{dev::Server, get, post, web::Json, App, HttpResponse, HttpServer, Responder};
use anyhow::Result;
use config::Servers;
use lumo::{
    agent::{Agent, FunctionCallingAgentBuilder},
    models::{openai::OpenAIServerModelBuilder, types::Message},
    tools::{
        AsyncTool, DuckDuckGoSearchTool, GoogleSearchTool, VisitWebsiteTool,
    },
};
#[cfg(feature = "code")]
use lumo::tools::PythonInterpreterTool;
#[cfg(feature = "mcp")]
use mcp_client::{
    ClientCapabilities, ClientInfo, McpClient, McpClientTrait, McpService, StdioTransport,
    Transport,
};
use serde::{Deserialize, Serialize};
use std::net::TcpListener;
use std::str::FromStr;
use actix_cors::Cors;
use actix_web::http::header;

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

#[get("/health_check")]
async fn health_check() -> impl Responder {
    HttpResponse::Ok()
}

#[post("/run")]
async fn run_task(
    req: Json<RunTaskRequest>,
) -> Result<impl Responder, actix_web::Error> {
    // use base url to get the right key from environment variables. if base url has openai, use openai key, if it has google, use google key, if it has groq, use groq key, if it has anthropic, use anthropic key
    let api_key = if req.base_url== "https://api.openai.com/v1/chat/completions" {
        std::env::var("OPENAI_API_KEY").ok()
    } else if req.base_url=="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions" {
        std::env::var("GOOGLE_API_KEY").ok()
    } else if req.base_url.to_lowercase().contains("groq") {
        std::env::var("GROQ_API_KEY").ok()
    } else if req.base_url.to_lowercase().contains("anthropic") {
        std::env::var("ANTHROPIC_API_KEY").ok()
    } else {
        None
    };

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

                let transport_handle = transport.start().await
                    .map_err(actix_web::error::ErrorInternalServerError)?;
                
                let service = McpService::new(transport_handle);
                let mut client = McpClient::new(service);

                client.initialize(
                    ClientInfo {
                        name: format!("{}-client", server_name),
                        version: "1.0.0".into(),
                    },
                    ClientCapabilities::default(),
                ).await.map_err(actix_web::error::ErrorInternalServerError)?;
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
                .await
                .map_err(actix_web::error::ErrorInternalServerError)?
        }
        _ => {
            // Default function calling agent logic...
            let servers = Servers::load().map_err(actix_web::error::ErrorInternalServerError)?;

            let tools = if let Some(tools) = &req.tools {
                tools.iter()
                    .map(|tool| ToolType::from_str(tool).map(|t| create_tool(&t)))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                vec![]
            };

            let mut agent =
                FunctionCallingAgentBuilder::new(model)
                    .with_tools(tools)
                    .with_max_steps(req.max_steps)
                    .with_history(req.history.clone())
                    .with_system_prompt(servers.system_prompt.as_deref())
                    .with_logging_level(Some(log::LevelFilter::Info))
                    .build()
                    .map_err(actix_web::error::ErrorInternalServerError)?;
            agent
                .run(&req.task, false)
                .await
                .map_err(actix_web::error::ErrorInternalServerError)?
        }
    };

    Ok(Json(RunTaskResponse { response }))
}

pub fn run(listener: TcpListener) -> std::io::Result<Server> {
    Ok(HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allowed_methods(vec!["GET", "POST"])
            .allowed_headers(vec![header::AUTHORIZATION, header::ACCEPT, header::CONTENT_TYPE])
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
