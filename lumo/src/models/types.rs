use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::openai::ToolCall;

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    System,
    #[serde(rename = "tool_calls")]
    ToolCall,
    #[serde(rename = "tool")]
    ToolResponse,
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageRole::User => write!(f, "User"),
            MessageRole::Assistant => write!(f, "Assistant"),
            MessageRole::System => write!(f, "System"),
            MessageRole::ToolCall => write!(f, "ToolCall"),
            MessageRole::ToolResponse => write!(f, "ToolResponse"),
        }
    }
}

#[derive(Debug, Serialize, Clone, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

pub struct MessageBuilder {
    role: MessageRole,
    content: String,
    tool_call_id: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

impl MessageBuilder {
    pub fn new(role: MessageRole, content: &str) -> Self {
        Self {
            role,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: None,
        }
    }
    pub fn with_tool_call_id(mut self, tool_call_id: &str) -> Self {
        self.tool_call_id = Some(tool_call_id.to_string());
        self
    }
    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }
    pub fn build(self) -> Message {
        Message {
            role: self.role,
            content: self.content,
            tool_call_id: self.tool_call_id,
            tool_calls: self.tool_calls,
        }
    }
}

impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Message(role: {}, content: {})", self.role, self.content)
    }
}

impl Message {
    pub fn new(role: MessageRole, content: &str) -> Self {
        Self {
            role,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: None,
        }
    }
}
