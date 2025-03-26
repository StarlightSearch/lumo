use lumo::agent::{Agent, FunctionCallingAgentBuilder};
use lumo::models::openai::OpenAIServerModelBuilder;
use lumo::tools::{AsyncTool, DuckDuckGoSearchTool, PythonInterpreterTool, VisitWebsiteTool};

#[tokio::main]
async fn main() {
    let tools: Vec<Box<dyn AsyncTool>> = vec![
        Box::new(DuckDuckGoSearchTool::new()),
        Box::new(VisitWebsiteTool::new()),
    ];
    let model = OpenAIServerModelBuilder::new("gpt-4o-mini")
        .with_base_url(Some("https://api.openai.com/v1/chat/completions"))
        .build()
        .unwrap();
 

    let coding_agent = FunctionCallingAgentBuilder::new(model.clone())
        .with_tools(vec![Box::new(PythonInterpreterTool::new())])
        .with_name(Some("coding_agent"))
        .with_logging_level(Some(log::LevelFilter::Info))
        .with_description(Some("A coding agent that can write code in python. Use this to do calculations and other complex tasks."))
        .build()
        .unwrap();
    let mut agent = FunctionCallingAgentBuilder::new(model)
        .with_tools(tools)
        .with_max_steps(Some(10))
        .with_logging_level(Some(log::LevelFilter::Info))
        .with_planning_interval(Some(1))
        .with_managed_agents(vec![Box::new(coding_agent) as Box<dyn Agent>])
        .build()
        .unwrap();

    let _result = agent.run("Find the stock price of Nvidia and the stock price of Apple and find the ratio of the two stock prices.", true).await.unwrap();
}
