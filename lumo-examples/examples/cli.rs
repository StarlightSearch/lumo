use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use clap::{Parser, ValueEnum};
use futures::StreamExt;
use lumo::agent::{AgentStream, CodeAgent, FunctionCallingAgent, StreamResult};
use lumo::agent::{CodeAgentBuilder, FunctionCallingAgentBuilder, Step};
use lumo::errors::AgentError;
use lumo::models::gemini::{GeminiServerModel, GeminiServerModelBuilder};
use lumo::models::model_traits::{Model, ModelResponse};
use lumo::models::ollama::{OllamaModel, OllamaModelBuilder};
use lumo::models::openai::{
    OpenAIServerModel, OpenAIServerModelBuilder, Status,
};
use lumo::models::types::Message;
use lumo::tools::{
    AsyncTool, DuckDuckGoSearchTool, ExaSearchTool, GoogleSearchTool, PythonInterpreterTool, TavilySearchTool, ToolInfo, VisitWebsiteTool
};
use serde_json;
use std::fs::File;
use tokio::sync::broadcast;

#[derive(Debug, Clone, ValueEnum)]
enum AgentType {
    FunctionCalling,
    Code,
}

#[derive(Debug, Clone, ValueEnum)]
enum ToolType {
    DuckDuckGo,
    VisitWebsite,
    GoogleSearchTool,
    PythonInterpreter,
    TavilySearch,
    ExaSearch,
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
    Gemini(GeminiServerModel),
}

enum AgentWrapper {
    FunctionCalling(FunctionCallingAgent<ModelWrapper>),
    Code(CodeAgent<ModelWrapper>),
}

impl AgentWrapper {
    fn stream_run<'a>(
        &'a mut self,
        task: &'a str,
        reset: bool,
        tx: Option<broadcast::Sender<Status>>,
    ) -> StreamResult<'a, Step> {
        match self {
            AgentWrapper::FunctionCalling(agent) => agent.stream_run(task, reset, tx),
            AgentWrapper::Code(agent) => agent.stream_run(task, reset, tx),
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
            ModelWrapper::Gemini(m) => {
                Ok(m.run(messages, history, tools, max_tokens, args).await?)
            }
        }
    }
    async fn run_stream(
        &self,
        messages: Vec<Message>,
        history: Option<Vec<Message>>,
        tools: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
        tx: broadcast::Sender<Status>,
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
        match self {
            ModelWrapper::OpenAI(m) => Ok(m
                .run_stream(messages, history, tools, max_tokens, args, tx)
                .await?),
            ModelWrapper::Ollama(m) => Ok(m
                .run_stream(messages, history, tools, max_tokens, args, tx)
                .await?),
            ModelWrapper::Gemini(m) => Ok(m
                .run_stream(messages, history, tools, max_tokens, args, tx)
                .await?),
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
    #[arg(short = 'm', long, value_enum, default_value = "open-ai")]
    model_type: ModelType,

    /// OpenAI API key (only required for OpenAI model)
    #[arg(short = 'k', long)]
    api_key: Option<String>,

    /// Model ID (e.g., "gpt-4" for OpenAI or "qwen2.5" for Ollama)
    #[arg(long, default_value = "gpt-4.1-mini")]
    model_id: String,

    /// Whether to reset the agent
    #[arg(short, long, default_value = "false")]
    reset: bool,

    /// The task to execute
    #[arg(short, long)]
    task: String,

    /// Base URL for the API
    #[arg(short, long)]
    base_url: Option<String>,

    /// Maximum number of steps to take
    #[arg(long, default_value = "10")]
    max_steps: Option<usize>,

    /// Enable streaming mode to see content as it's generated
    #[arg(short, long, default_value = "false")]
    stream: bool,
}

fn create_tool(tool_type: &ToolType) -> Box<dyn AsyncTool> {
    match tool_type {
        ToolType::DuckDuckGo => Box::new(DuckDuckGoSearchTool::new()),
        ToolType::VisitWebsite => Box::new(VisitWebsiteTool::new()),
        ToolType::GoogleSearchTool => Box::new(GoogleSearchTool::new(None)),
        ToolType::PythonInterpreter => Box::new(PythonInterpreterTool::new()),
        ToolType::TavilySearch => Box::new(TavilySearchTool::new(None)),
        ToolType::ExaSearch => Box::new(ExaSearchTool::new(10, None)),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let tools: Vec<Box<dyn AsyncTool>> = args.tools.iter().map(create_tool).collect();

    // Create model based on type
    let model = match args.model_type {
        ModelType::OpenAI => ModelWrapper::OpenAI(
            OpenAIServerModelBuilder::new(&args.model_id)
                .with_base_url(args.base_url.as_deref())
                .with_api_key(args.api_key.as_deref())
                .build()?,
        ),
        ModelType::Gemini => ModelWrapper::Gemini(
            GeminiServerModelBuilder::new(&args.model_id)
                .with_base_url(args.base_url.as_deref())
                .with_api_key(args.api_key.as_deref())
                .build()?,
        ),
        ModelType::Ollama => ModelWrapper::Ollama(
            OllamaModelBuilder::new()
                .model_id(&args.model_id)
                .ctx_length(8000)
                .temperature(Some(0.0))
                .with_native_tools(true)
                .build(),
        ),
    };

    // Create agent based on type
    let mut agent = match args.agent_type {
        AgentType::FunctionCalling => AgentWrapper::FunctionCalling(
            FunctionCallingAgentBuilder::new(model)
                .with_tools(tools)
                .with_system_prompt(Some("CLI Agent"))
                .with_max_steps(args.max_steps)
                .build()?,
        ),
        AgentType::Code => AgentWrapper::Code(
            CodeAgentBuilder::new(model)
                .with_tools(tools)
                .with_system_prompt(Some("CLI Agent"))
                .with_max_steps(args.max_steps)
                .build()?,
        ),
    };

    // Store logs in a file
    let mut file = File::create("logs.txt")?;

    // Setup streaming channel if needed
    let tx = if args.stream {
        let (tx, mut rx) = broadcast::channel::<Status>(100);
        
        // Spawn a non-blocking task to handle streaming status messages
        tokio::spawn(async move {
            while let Ok(status) = rx.recv().await {
                match status {
                    Status::Content(content) => {
                        use std::io::Write;
                        print!("{}", content);
                        let _ = std::io::stdout().flush();
                    }
                    _ => {}
                }
            }
        });
        
        Some(tx)
    } else {
        None
    };

    // Run the agent with streaming
    let mut result = agent.stream_run(&args.task, args.reset, tx)?;

    // Process the stream and collect results
    while let Some(step) = result.next().await {
        if let Ok(step) = step {
            // Write to log file
            serde_json::to_writer_pretty(&mut file, &step)?;
            
            // Print step information to console
            match &step {
                Step::ActionStep(agent_step) => {
                    if let Some(tool_calls) = &agent_step.tool_call {
                        for tool_call in tool_calls {
                            println!("\n🔧 Tool: {} | Arguments: {}", tool_call.function.name, tool_call.function.arguments);
                        }
                        // Add spacing after tool calls in streaming mode
                        if args.stream {
                            println!();
                        }
                    }
                    if let Some(answer) = &agent_step.final_answer {
                        // For streaming mode, content is already being printed
                        // For non-streaming mode, print the final answer here
                        if !args.stream {
                            println!("\n=== Final Answer ===");
                            println!("{}", answer);
                        } else {
                            println!("\n"); // Add newline after streamed content
                        }
                    }
                }
                Step::ToolCall(tool_call) => {
                    println!("\n🔧 Tool: {} | Arguments: {}", tool_call.function.name, tool_call.function.arguments);
                    // Add spacing after tool calls in streaming mode
                    if args.stream {
                        println!();
                    }
                }
                Step::PlanningStep(plan, facts) => {
                    println!("\n📋 Planning Step");
                    println!("Plan: {}", plan);
                    println!("Facts: {}", facts);
                }
                Step::TaskStep(task) => {
                    println!("\n📝 Task: {}", task);
                }
                _ => {}
            }
        } else {
            eprintln!("Error: {:?}", step);
        }
    }

    if !args.stream {
        println!("\n=== Task Completed ===");
        println!("Logs saved to logs.txt");
    }

    Ok(())
}

