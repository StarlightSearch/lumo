use anyhow::Result;
use async_trait::async_trait;
use clap::{Parser, ValueEnum};
use futures::StreamExt;
use lumo::agent::{
    AgentStream, CodeAgent, CodeAgentBuilder, FunctionCallingAgent, FunctionCallingAgentBuilder,
    McpAgentBuilder, StreamResult,
};
use lumo::agent::{McpAgent, Step};
use lumo::errors::AgentError;
use lumo::models::model_traits::{Model, ModelResponse};
use lumo::models::ollama::{OllamaModel, OllamaModelBuilder};
use lumo::models::openai::{OpenAIServerModel, OpenAIServerModelBuilder};
use lumo::models::types::Message;
use lumo::tools::{
    AsyncTool, DuckDuckGoSearchTool, GoogleSearchTool, PythonInterpreterTool, ToolInfo,
    VisitWebsiteTool,
};
use mcp_client::{
    ClientCapabilities, ClientInfo, McpClient, McpClientTrait, McpService, StdioTransport,
    Transport,
};
use mcp_core::protocol::JsonRpcMessage;
use opentelemetry::trace::{FutureExt, SpanKind, TraceContextExt, Tracer};
use opentelemetry::{global, Context, KeyValue};
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::time::Duration;
use tower::Service;
use tracing::Level;
use tracing_subscriber::{fmt, EnvFilter};
mod config;
use config::Servers;
mod cli_utils;
use cli_utils::{CliPrinter, ToolCallsFormatter};
mod splash;
use splash::SplashScreen;
mod telemetry;
use telemetry::init_tracer;

#[derive(Debug, Clone, ValueEnum)]
enum AgentType {
    FunctionCalling,
    Code,
    Mcp,
}

#[derive(Debug, Clone, ValueEnum)]
enum ToolType {
    DuckDuckGo,
    VisitWebsite,
    GoogleSearchTool,
    PythonInterpreter,
}

#[derive(Debug, Clone, ValueEnum)]
enum ModelType {
    OpenAI,
    Ollama,
    Gemini,
}

#[derive(Debug)]
enum ModelWrapper {
    OpenAI(OpenAIServerModel),
    Ollama(OllamaModel),
}

enum AgentWrapper<
    M: Model + Send + Sync + std::fmt::Debug + 'static,
    S: Service<JsonRpcMessage, Response = JsonRpcMessage> + Clone + Send + Sync + 'static,
> where
    S::Future: Send,
    S::Error: Into<mcp_client::Error>,
{
    FunctionCalling(FunctionCallingAgent<M>),
    Code(CodeAgent<M>),
    Mcp(McpAgent<M, S>),
}

impl<
        M: Model + Send + Sync + std::fmt::Debug + 'static,
        S: Service<JsonRpcMessage, Response = JsonRpcMessage> + Clone + Send + Sync + 'static,
    > AgentWrapper<M, S>
where
    S::Future: Send,
    S::Error: Into<mcp_client::Error>,
{
    fn stream_run<'a>(&'a mut self, task: &'a str, reset: bool) -> StreamResult<'a, Step> {
        match self {
            AgentWrapper::FunctionCalling(agent) => agent.stream_run(task, reset),
            AgentWrapper::Code(agent) => agent.stream_run(task, reset),
            AgentWrapper::Mcp(agent) => agent.stream_run(task, reset),
        }
    }
}

#[async_trait]
impl Model for ModelWrapper {
    async fn run(
        &self,
        messages: Vec<Message>,
        history: Option<Vec<Message>>,
        tools: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
        match self {
            ModelWrapper::OpenAI(m) => {
                Ok(m.run(messages, history, tools, max_tokens, args).await?)
            }
            ModelWrapper::Ollama(m) => {
                Ok(m.run(messages, history, tools, max_tokens, args).await?)
            }
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The type of agent to use
    #[arg(short = 'a', long, value_enum, default_value = "function-calling")]
    agent_type: AgentType,

    /// List of tools to use
    #[arg(short = 'l', long = "tools", value_enum, num_args = 1.., value_delimiter = ',', default_values_t = [ToolType::DuckDuckGo, ToolType::VisitWebsite])]
    tools: Vec<ToolType>,

    /// The type of model to use
    #[arg(short = 'm', long, value_enum, default_value = "gemini")]
    model_type: ModelType,

    /// OpenAI API key (only required for OpenAI model)
    #[arg(short = 'k', long)]
    api_key: Option<String>,

    /// Model ID (e.g., "gpt-4" for OpenAI or "qwen2.5" for Ollama)
    #[arg(long, default_value = "gemini-2.0-flash")]
    model_id: String,

    /// Base URL for the API
    #[arg(short, long)]
    base_url: Option<String>,

    /// Maximum number of steps to take
    #[arg(long, default_value = "10")]
    max_steps: Option<usize>,

    /// Planning interval
    #[arg(short = 'p', long)]
    planning_interval: Option<usize>,

    /// Logging level
    #[arg(short = 'v', long)]
    logging_level: Option<log::LevelFilter>,
}

fn create_tool(tool_type: &ToolType) -> Box<dyn AsyncTool> {
    match tool_type {
        ToolType::DuckDuckGo => Box::new(DuckDuckGoSearchTool::new()),
        ToolType::VisitWebsite => Box::new(VisitWebsiteTool::new()),
        ToolType::GoogleSearchTool => Box::new(GoogleSearchTool::new(None)),
        ToolType::PythonInterpreter => Box::new(PythonInterpreterTool::new()),
    }
}

#[tracing::instrument]
#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize tracing subscriber with custom formatting
    let tracer_provider = init_tracer();
    let (tracer, cx) = if let Some(_) = &tracer_provider {
        let tracer = global::tracer("lumo");
        let span = tracer
            .span_builder("conversation")
            .with_kind(SpanKind::Internal)
            .with_start_time(std::time::SystemTime::now())
            .with_attributes(vec![
                KeyValue::new("gen_ai.operation.name", "conversation"),
                KeyValue::new(
                    "gen_ai.base_url",
                    args.base_url.clone().unwrap_or(
                        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
                            .to_string(),
                    ),
                ),
                KeyValue::new("timestamp", chrono::Utc::now().to_rfc3339()),
            ])
            .start(&tracer);
        (Some(tracer), Some(Context::current_with_span(span)))
    } else {
        (None, None)
    };

    let subscriber = fmt::Subscriber::builder()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive(Level::INFO.into())
                .add_directive("lumo=debug".parse().unwrap()),
        )
        .with_writer(io::stdout)
        .event_format(ToolCallsFormatter)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("Failed to set subscriber");

    // Display splash screen
    let config_path = Servers::config_path()?;
    let servers = Servers::load()?;

    let endpoint = if let Some((_, endpoint)) = &tracer_provider {
        Some(endpoint.clone())
    } else {
        None
    };

    SplashScreen::display(
        &config_path,
        &servers.servers.keys().cloned().collect::<Vec<_>>(),
        &args.model_id,
        endpoint,
    );

    let tools: Vec<Box<dyn AsyncTool>> = args.tools.iter().map(create_tool).collect();

    // Create model based on type
    let model = match args.model_type {
        ModelType::OpenAI => ModelWrapper::OpenAI(
            OpenAIServerModelBuilder::new(&args.model_id)
                .with_base_url(args.base_url.as_deref())
                .with_api_key(args.api_key.as_deref())
                .build()?,
        ),
        ModelType::Gemini => ModelWrapper::OpenAI(
            OpenAIServerModelBuilder::new(&args.model_id)
                .with_base_url(Some(args.base_url.as_deref().unwrap_or(
                    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                )))
                .with_api_key(Some(
                    args.api_key.as_deref().unwrap_or(
                        &std::env::var("GOOGLE_API_KEY")
                            .unwrap_or_else(|_| "Gemini API key not found".to_string()),
                    ),
                ))
                .build()?,
        ),
        ModelType::Ollama => ModelWrapper::Ollama(
            OllamaModelBuilder::new()
                .model_id(&args.model_id)
                .ctx_length(20000)
                .temperature(Some(0.1))
                .url(args.base_url.as_deref().unwrap_or("http://localhost:11434"))
                .with_native_tools(true)
                .build(),
        ),
    };

    let system_prompt = match args.model_type {
        ModelType::Ollama => Some(
            r#"You are a helpful assistant that can answer questions and help with tasks. You are given access tools which you can use to answer the user's question. 
        
1. You can use multiple tools to answer the user's question.
2. Do not use the tool with the same parameters more than once.
3. Provide a detailed response in a well structured and easy to understand manner.
4. If you don't have enough information to answer the user's question, say so.
5. When needed, provide references to the sources you used to answer the user's question. You can provide these references in a list format at the end of your response.

The current time is {{current_time}}"#,
        ),
        _ => servers.system_prompt.as_deref(),
    };

    let mut agent = match args.agent_type {
        AgentType::FunctionCalling => AgentWrapper::FunctionCalling(
            FunctionCallingAgentBuilder::new(model)
                .with_tools(tools)
                .with_system_prompt(system_prompt)
                .with_max_steps(args.max_steps)
                .with_planning_interval(args.planning_interval)
                .with_logging_level(args.logging_level)
                .build()?,
        ),
        AgentType::Code => AgentWrapper::Code(
            CodeAgentBuilder::new(model)
                .with_tools(tools)
                .with_max_steps(args.max_steps)
                .with_planning_interval(args.planning_interval)
                .with_logging_level(args.logging_level)
                .build()?,
        ),
        AgentType::Mcp => {
            // Initialize all configured servers
            let mut clients = Vec::new();
            let servers = Servers::load()?;
            // Iterate through all server configurations
            for (server_name, server_config) in servers.servers.iter() {
                // Create transport for this server
                let transport = StdioTransport::new(
                    &server_config.command,
                    server_config.args.clone(),
                    server_config.env.clone().unwrap_or_default(),
                );

                // Start the transport
                let transport_handle = transport.start().await?;
                let service = McpService::with_timeout(transport_handle, Duration::from_secs(10));
                let mut client = McpClient::new(service);

                // Initialize the client with a unique name
                let _ = client
                    .initialize(
                        ClientInfo {
                            name: format!("{}-client", server_name),
                            version: "1.0.0".into(),
                        },
                        ClientCapabilities::default(),
                    )
                    .await?;

                clients.push(client);
            }

            // Create MCP agent with all initialized clients
            AgentWrapper::Mcp(
                McpAgentBuilder::new(model)
                    .with_system_prompt(system_prompt)
                    .with_max_steps(args.max_steps)
                    .with_planning_interval(args.planning_interval)
                    .with_mcp_clients(clients)
                    .build()
                    .await?,
            )
        }
    };

    let mut file: File = File::create("logs.txt")?;

    let mut task_count = 1;
    loop {
        let mut cli_printer = CliPrinter::new()?;
        let task = cli_printer.prompt_user()?;

        let task_name = format!("Task {}", task_count);

        if task.is_empty() {
            CliPrinter::handle_empty_input();
            continue;
        }
        if task == "exit" {
            if let (Some((provider, _)), Some(context)) = (&tracer_provider, &cx) {
                context.span().end();
                // Ensure all spans are exported before shutting down
                provider.force_flush()?;
                provider.shutdown()?;
            }
            CliPrinter::print_goodbye();
            break;
        }
        let cx2 = if let (Some(t), Some(context)) = (&tracer, &cx) {
            let span = t
                .span_builder(task_name)
                .with_kind(SpanKind::Internal)
                .with_start_time(std::time::SystemTime::now())
                .with_attributes(vec![
                    KeyValue::new("gen_ai.operation.name", "task"),
                    KeyValue::new("input.value", task.clone()),
                ])
                .start_with_context(t, context);
            Some(Context::current_with_span(span))
        } else {
            None
        };

        let mut result = agent.stream_run(&task, false)?;
        let mut final_answer = String::new();
        while let Some(step) = if let Some(context) = &cx2 {
            result.next().with_context(context.clone()).await
        } else {
            result.next().await
        } {
            if let Ok(step) = step {
                serde_json::to_writer_pretty(&mut file, &step)?;
                let answer = CliPrinter::print_step(&step)?;
                final_answer = answer;
            } else {
                println!("Error: {:?}", step);
            }
        }
        if let Some(context) = &cx2 {
            context.span().set_attribute(KeyValue::new("output.value", final_answer));
            context.span().end();
        }
        task_count += 1;
    }

    Ok(())
}
