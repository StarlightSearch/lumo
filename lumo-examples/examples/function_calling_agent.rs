use lumo::agent::{Agent, FunctionCallingAgentBuilder};
use lumo::models::openai::OpenAIServerModelBuilder;
use lumo::tools::{AsyncTool, DuckDuckGoSearchTool, VisitWebsiteTool};

#[tokio::main]
async fn main() {
    let tools: Vec<Box<dyn AsyncTool>> = vec![
        Box::new(DuckDuckGoSearchTool::new()),
        Box::new(VisitWebsiteTool::new()),
    ];
    let model = OpenAIServerModelBuilder::new("gpt-4o-mini")
        .with_base_url(Some(
            "https://api.openai.com/v1/chat/completions",
        ))
        .build()
        .unwrap();
    let mut agent = FunctionCallingAgentBuilder::new(model)
        .with_tools(tools)
        .with_system_prompt(Some(
            "You are a helpful assistant that can answer questions and help with tasks.",
        ))
        .with_max_steps(Some(10))
        .with_logging_level(Some(log::LevelFilter::Info))
        .with_planning_interval(Some(1))
        .build()
        .unwrap();
    let _result = agent.run("who is elon musk", true).await.unwrap();
}
