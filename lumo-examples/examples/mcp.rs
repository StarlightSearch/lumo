use std::collections::HashMap;

use anyhow::Result;
use mcp_client::{
    ClientCapabilities, ClientInfo, Error as ClientError, McpClient, McpClientTrait, McpService,
    StdioTransport, Transport,
};
use lumo::agent::{Agent, McpAgentBuilder};
use lumo::models::openai::OpenAIServerModelBuilder;
use std::time::Duration;

use lumo::agent::mcp_agent::McpAgent;
use lumo::prompts::TOOL_CALLING_SYSTEM_PROMPT;

#[tokio::main]
async fn main() -> Result<(), ClientError> {


    // 1) Create the transport
    let transport = StdioTransport::new(
        "npx",
        vec![
            "@modelcontextprotocol/server-filesystem".to_string(),
            "/home/akshay/projects/smolagents-rs".to_string(),
        ],
        HashMap::new(),
    );

    // 2) Start the transport to get a handle
    let transport_handle = transport.start().await?;

    // 3) Create the service with timeout middleware
    let service = McpService::with_timeout(transport_handle, Duration::from_secs(10));

    // 4) Create the client with the middleware-wrapped service
    let mut client = McpClient::new(service);

    // Initialize
    client
        .initialize(
            ClientInfo {
                name: "test-client".into(),
                version: "1.0.0".into(),
            },
            ClientCapabilities::default(),
        )
        .await?;

    let model = OpenAIServerModelBuilder::new("gpt-4o-mini")
        .with_base_url(Some("https://api.openai.com/v1/chat/completions"))
        .build()
        .unwrap();

    let mut agent = McpAgentBuilder::new(model)
        .with_system_prompt(Some(TOOL_CALLING_SYSTEM_PROMPT))
        .with_mcp_clients(vec![client])
        .build()
        .await.unwrap();
    // Use agent here
    let _result = agent.run("List the directories in the current directory!", false).await.unwrap();

    Ok(())
}
