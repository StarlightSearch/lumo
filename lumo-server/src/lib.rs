pub mod config;
use actix_web::{dev::Server, get, post, web::{self, Json}, App, HttpResponse, HttpServer, Responder};
use anyhow::Result;
use config::Servers;
use lumo::{
    agent::{Agent,FunctionCallingAgentBuilder, McpAgentBuilder},
    models::{openai::OpenAIServerModelBuilder, types::Message},
    tools::{
        AsyncTool, DuckDuckGoSearchTool, GoogleSearchTool, PythonInterpreterTool, VisitWebsiteTool,
    },
};
use mcp_client::{
    ClientCapabilities, ClientInfo, McpClient, McpClientTrait, McpService, StdioTransport,
    Transport,
};
use mcp_client::transport::stdio::StdioTransportHandle;
use serde::{Deserialize, Serialize};
use std::net::TcpListener;
use std::str::FromStr;
use actix_cors::Cors;
use actix_web::http::header;
use std::sync::Arc;
use tokio::sync::Mutex;


#[derive(Deserialize)]
struct RunTaskRequest {
    task: String,
    model: String,
    base_url: String,
    tools: Vec<String>,
    api_key: Option<String>,
    max_steps: Option<usize>,
    history: Option<Vec<Message>>,
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
    PythonInterpreter,
}

impl FromStr for ToolType {
    type Err = actix_web::error::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "DuckDuckGo" => Ok(ToolType::DuckDuckGo),
            "VisitWebsite" => Ok(ToolType::VisitWebsite),
            "GoogleSearchTool" => Ok(ToolType::GoogleSearchTool),
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
        ToolType::PythonInterpreter => Box::new(PythonInterpreterTool::new()),
    }
}

#[get("/health_check")]
async fn health_check() -> impl Responder {
    HttpResponse::Ok()
}

// Create a type to hold our initialized MCP clients
struct AppState {
    mcp_clients: Arc<Mutex<Vec<McpClient<McpService<StdioTransportHandle>>>>>,
}

#[post("/run")]
async fn run_task(
    req: Json<RunTaskRequest>,
) -> Result<impl Responder, actix_web::Error> {
    let model = OpenAIServerModelBuilder::new(&req.model)
        .with_base_url(Some(&req.base_url))
        .with_api_key(req.api_key.as_deref())
        .build()
        .map_err(actix_web::error::ErrorInternalServerError)?;

    let response = match req.agent_type.as_deref() {
        Some("mcp") => {
            // Create fresh clients for this request
            let mut clients = Vec::new();
            let servers = Servers::load().map_err(actix_web::error::ErrorInternalServerError)?;
            
            for (server_name, server_config) in servers.servers.iter() {
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
            
            // Create and run MCP agent with fresh clients
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
            let tools = req
                .tools
                .iter()
                .map(|tool| ToolType::from_str(tool).map(|t| create_tool(&t)))
                .collect::<Result<Vec<_>, _>>()?;

            let mut agent =
                FunctionCallingAgentBuilder::new(model)
                    .with_tools(tools)
                    .with_max_steps(req.max_steps)
                    .with_history(req.history.clone())
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
            .allowed_origin("http://localhost:5173")
            .allowed_methods(vec!["GET", "POST"])
            .allowed_headers(vec![header::AUTHORIZATION, header::ACCEPT, header::CONTENT_TYPE])
            .max_age(3600);

        App::new()
            .wrap(cors)
            .service(health_check)
            .service(run_task)
    })
    .listen(listener)?
    .run())
}
