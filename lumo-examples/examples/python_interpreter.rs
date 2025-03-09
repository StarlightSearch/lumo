use lumo::agent::{Agent, FunctionCallingAgentBuilder};
use lumo::models::openai::OpenAIServerModelBuilder;
use lumo::tools::{AsyncTool, PythonInterpreterTool};

#[tokio::main]
async fn main() {
    let tools: Vec<Box<dyn AsyncTool>> = vec![
        Box::new(PythonInterpreterTool::new()),
    ];
    let model = OpenAIServerModelBuilder::new("gpt-4o-mini")
        .with_base_url(Some("https://api.openai.com/v1/chat/completions".to_string()))
        .build()
        .unwrap();
    let mut agent = FunctionCallingAgentBuilder::new(model)
        .with_tools(tools)
        .with_system_prompt(Some("You are a helpful assistant that can answer questions and help with tasks."))
        .with_max_steps(Some(10))
        .build()
        .unwrap();
    let _result = agent
        .run(
            "What is 2 + 2?",
            true,
        )
        .await
        .unwrap();
    println!("Result: {}", _result);

}
