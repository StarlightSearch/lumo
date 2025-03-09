use lumo::agent::{Agent, FunctionCallingAgentBuilder};
use lumo::models::gemini::GeminiServerModelBuilder;
use lumo::tools::{AsyncTool, DuckDuckGoSearchTool, VisitWebsiteTool};

#[tokio::main]
async fn main() {
    let tools: Vec<Box<dyn AsyncTool>> = vec![
        Box::new(DuckDuckGoSearchTool::new()),
        Box::new(VisitWebsiteTool::new()),
    ];
    let model = GeminiServerModelBuilder::new("gemini-2.0-flash")
        .build()
        .unwrap();

    let mut agent = FunctionCallingAgentBuilder::new(model)
        .with_tools(tools)
        .with_system_prompt(Some(
            "You are a helpful assistant that can answer questions and help with tasks.",
        ))
        .with_max_steps(Some(10))
        .with_planning_interval(Some(1))
        .with_logging_level(Some(log::LevelFilter::Info))
        .build()
        .unwrap();
    let _result = agent.run("who is elon musk", true).await.unwrap();
}
