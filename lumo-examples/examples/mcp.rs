use anyhow::Result;
use log::LevelFilter;
use lumo::agent::{Agent, McpAgentBuilder};
use lumo::models::openai::OpenAIServerModelBuilder;
use rmcp::{
    transport::{ConfigureCommandExt, TokioChildProcess},
    ServiceExt,
};
use std::time::Duration;
use tokio::process::Command;

use lumo::prompts::TOOL_CALLING_SYSTEM_PROMPT;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = ()
        .serve(TokioChildProcess::new(Command::new("npx").configure(|cmd| {
            cmd.args(&[
            "@modelcontextprotocol/server-filesystem",
            "/home/akshay/projects/smolagents-rs",
        ]);
        }))?)
        .await?;

    let model = OpenAIServerModelBuilder::new("gpt-4o-mini")
        .with_base_url(Some("https://api.openai.com/v1/chat/completions"))
        .build()
        .unwrap();

    let mut agent = McpAgentBuilder::new(model)
        .with_system_prompt(Some(TOOL_CALLING_SYSTEM_PROMPT))
        .with_mcp_clients(vec![client])
        .with_logging_level(Some(LevelFilter::Info))
        .build()
        .await
        .unwrap();
    // Use agent here
    let _result = agent
        .run("List the directories in the current directory!", false)
        .await
        .unwrap();

    Ok(())
}
