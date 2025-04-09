use lumo::agent::{Agent, FunctionCallingAgentBuilder};
use lumo::models::openai::OpenAIServerModelBuilder;
use lumo::tools::{
    AsyncTool, DuckDuckGoSearchTool, GoogleSearchTool, PythonInterpreterTool, VisitWebsiteTool,
};

#[tokio::main]
async fn main() {
    let tools: Vec<Box<dyn AsyncTool>> = vec![
        Box::new(GoogleSearchTool::new(None)),
        Box::new(VisitWebsiteTool::new()),
    ];
    let model = OpenAIServerModelBuilder::new("gpt-4o-mini")
        .with_base_url(Some("https://api.openai.com/v1/chat/completions"))
        .build()
        .unwrap();

    let agent = FunctionCallingAgentBuilder::new(model.clone())
        .with_tools(tools)
        .with_name(Some("search_agent"))
        .with_max_steps(Some(10))
        .with_logging_level(Some(log::LevelFilter::Info))
        .with_planning_interval(Some(1))
        .build()
        .unwrap();

    let mut coding_agent = FunctionCallingAgentBuilder::new(model)
        .with_tools(vec![Box::new(PythonInterpreterTool::new())])
        .with_name(Some("coding_agent"))
        .with_logging_level(Some(log::LevelFilter::Info))
        .with_description(Some("A coding agent that can write code in python. Use this to do calculations and other complex tasks."))
        .with_managed_agents(vec![Box::new(agent) as Box<dyn Agent>])
        .build()
        .unwrap();

    let _result = coding_agent.run("Find the stock price of Nvidia and the stock price of Apple and find the ratio of the two stock prices.", true).await.unwrap();
}
