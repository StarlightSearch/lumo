use std::collections::HashMap;

use crate::{
    errors::AgentError,
    models::{
        openai::{Status, ToolCall},
        types::Message,
    },
    tools::tool_traits::ToolInfo,
};
use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::broadcast;

pub trait ModelResponse: Send + Sync {
    fn get_response(&self) -> Result<String, AgentError>;
    fn get_tools_used(&self) -> Result<Vec<ToolCall>, AgentError>;
}

#[async_trait]
pub trait Model: Send + Sync + 'static {
    async fn run(
        &self,
        input_messages: Vec<Message>,
        history: Option<Vec<Message>>,
        tools: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
    ) -> Result<Box<dyn ModelResponse>, AgentError>;

    async fn run_stream(
        &self,
        input_messages: Vec<Message>,
        history: Option<Vec<Message>>,
        tools: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
        tx: broadcast::Sender<Status>,
    ) -> Result<Box<dyn ModelResponse>, AgentError>;
}
