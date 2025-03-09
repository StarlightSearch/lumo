use actix_web::{dev::Server, get, post, web::Json, App, HttpResponse, HttpServer, Responder};
use anyhow::Result;
use lumo::{
    agent::{Agent,FunctionCallingAgentBuilder},
    models::{openai::OpenAIServerModelBuilder, types::Message},
    tools::{
        AsyncTool, DuckDuckGoSearchTool, GoogleSearchTool, PythonInterpreterTool, VisitWebsiteTool,
    },
};
use serde::{Deserialize, Serialize};
use std::{net::TcpListener, str::FromStr};
use actix_cors::Cors;
use actix_web::http::header;

#[derive(Deserialize)]
struct RunTaskRequest {
    task: String,
    model: String,
    base_url: String,
    tools: Vec<String>,
    api_key: Option<String>,
    max_steps: Option<usize>,
    history: Option<Vec<Message>>,
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

#[post("/run")]
async fn run_task(req: Json<RunTaskRequest>) -> Result<impl Responder, actix_web::Error> {
    let model = OpenAIServerModelBuilder::new(&req.model)
        .with_base_url(Some(req.base_url.clone()))
        .with_api_key(req.api_key.clone())
        .build()
        .map_err(actix_web::error::ErrorInternalServerError)?;
    let tools = req
        .tools
        .iter()
        .map(|tool| ToolType::from_str(tool).map(|t| create_tool(&t)))
        .collect::<Result<Vec<_>, _>>()?;

    let mut agent =
        FunctionCallingAgentBuilder::new(model)
            .with_tools(tools)
            .with_system_prompt(Some("CLI Agent"))
            .with_max_steps(req.max_steps)
            .with_history(req.history.clone())
            .with_logging_level(Some(log::LevelFilter::Info))
            .build()
            .map_err(actix_web::error::ErrorInternalServerError)?;
    let response = agent
        .run(&req.task, false)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;
    Ok(Json(RunTaskResponse { response }))
}

pub fn run(listener: TcpListener) -> std::io::Result<Server> {
    Ok(
        HttpServer::new(|| {
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
            .run(),
    )
}
