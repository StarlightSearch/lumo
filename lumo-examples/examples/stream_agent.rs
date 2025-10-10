use std::io::Write;

use anyhow::Result;
use lumo::{
    models::{
        model_traits::Model,
        openai::{OpenAIServerModelBuilder, Status},
        types::{Message, MessageRole},
    },
    tools::{AnyTool, DuckDuckGoSearchTool},
};
use tokio::sync::broadcast;
#[tokio::main]
async fn main() -> Result<()> {
    let model = OpenAIServerModelBuilder::new("gpt-4.1-mini")
        .with_base_url(Some("https://api.openai.com/v1/chat/completions"))
        .build()
        .unwrap();

    let prompt = "What are patch embeddings?  and what is the capital of France? Use multiple tools at the same time to answer the question.";
    let tool = DuckDuckGoSearchTool::new().tool_info();

    let (tx, mut rx) = broadcast::channel::<Status>(32);
    let accumulated_response = model
        .run_stream(
            vec![Message::new(MessageRole::User, prompt)],
            None,
            vec![tool],
            None,
            None,
            tx,
        )
        .await?;

    println!("Starting broadcast pattern test...");

    // Process UI stream
    let mut ui_content = String::new();
    while let Ok(content) = rx.recv().await {
        match content {
            Status::FirstContent(content) => {
                println!("Final Answer: {}", content);
                ui_content.push_str(&content);
                print!("{}", content);
            }
            Status::Content(content) => {
                ui_content.push_str(&content);
                print!("{}", content);
            }
            _ => {}
        }
        std::io::stdout().flush().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    println!();

    // let accumulated_content = accumulated_response;
    // println!(
    //     "Accumulated content: {:?}",
    //     accumulated_content.get_tools_used().unwrap()
    // );
    // println!("UI content: {}", ui_content);

    Ok(())
}
