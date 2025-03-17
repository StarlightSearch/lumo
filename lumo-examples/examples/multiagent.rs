use std::collections::HashMap;

use lumo::agent::{Agent, FunctionCallingAgentBuilder};
use lumo::models::openai::OpenAIServerModelBuilder;
use lumo::tools::{AsyncTool, DuckDuckGoSearchTool, PythonInterpreterTool, VisitWebsiteTool};

#[tokio::main]
async fn main() {
    let tools: Vec<Box<dyn AsyncTool>> = vec![
        Box::new(DuckDuckGoSearchTool::new()),
        Box::new(VisitWebsiteTool::new()),
    ];
    let model1 = OpenAIServerModelBuilder::new("gpt-4o-mini")
        .with_base_url(Some("https://api.openai.com/v1/chat/completions"))
        .build()
        .unwrap();
    let model2 = OpenAIServerModelBuilder::new("gpt-4o-mini")
        .with_base_url(Some("https://api.openai.com/v1/chat/completions"))
        .build()
        .unwrap();

    let coding_agent = FunctionCallingAgentBuilder::new(model2)
        .with_tools(vec![Box::new(PythonInterpreterTool::new())])
        .with_description(Some("A coding agent that can write code in python"))
        .build()
        .unwrap();
    let mut agent = FunctionCallingAgentBuilder::new(model1)
        .with_tools(tools)
        .with_max_steps(Some(10))
        .with_logging_level(Some(log::LevelFilter::Info))
        .with_planning_interval(Some(1))
        .with_managed_agents(Some(HashMap::from([(
            "coding_agent".to_string(),
            Box::new(coding_agent) as Box<dyn Agent>,
        )])))
        .build()
        .unwrap();

    let _result = agent.run("who is elon musk", true).await.unwrap();
}
