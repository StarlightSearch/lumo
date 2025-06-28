use std::collections::HashMap;

use crate::{
    errors::AgentError,
    models::{
        openai::Status,
        types::{Message, MessageRole},
    },
    tools::ToolInfo,
};
use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::broadcast;

use super::{
    model_traits::{Model, ModelResponse},
    openai::{FunctionCall, ToolCall},
};

/// Text content within a chat message
#[derive(Serialize, Debug)]
#[serde(rename_all = "snake_case")]
enum GeminiContentPart {
    /// The actual text content
    Text(String),
}

#[derive(Serialize, Debug)]
struct GeminiChatContent {
    role: String,
    parts: Vec<GeminiContentPart>,
}

#[derive(Serialize)]
struct GeminiTool {
    function_declarations: Vec<Value>,
}
#[derive(Serialize)]
struct GeminiChatRequest {
    contents: Vec<GeminiChatContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<GeminiTool>,
    generation_config: GeminiGenerationConfig,
}

/// Configuration parameters for text generation
#[derive(Serialize)]
struct GeminiGenerationConfig {
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    /// Sampling temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    /// Top-p sampling parameter
    #[serde(skip_serializing_if = "Option::is_none", rename = "topP")]
    top_p: Option<f32>,
    /// Top-k sampling parameter
    #[serde(skip_serializing_if = "Option::is_none", rename = "topK")]
    top_k: Option<u32>,
    /// Stop sequences
    #[serde(rename = "stopSequences", skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

/// Individual completion candidate
#[derive(Deserialize, Debug)]
struct GeminiCandidate {
    /// Content of the candidate response
    content: GeminiResponseContent,
}

/// Content block within a response
#[derive(Deserialize, Debug)]
struct GeminiResponseContent {
    /// Parts making up the content
    parts: Vec<GeminiResponsePart>,
}

#[derive(Deserialize, Debug, Clone)]
struct GeminiFunctionCall {
    name: String,
    args: Value,
}

/// Individual part of response content
#[derive(Deserialize, Debug)]
struct GeminiResponsePart {
    /// Text content of this part
    text: Option<String>,
    /// Tool calls
    #[serde(rename = "functionCall")]
    function_call: Option<GeminiFunctionCall>,
}

/// Response from the chat completion API
#[derive(Deserialize, Debug)]
struct GeminiChatResponse {
    /// Generated completion candidates
    candidates: Vec<GeminiCandidate>,
}

impl ModelResponse for GeminiChatResponse {
    fn get_response(&self) -> Result<String, AgentError> {
        Ok(self.candidates[0].content.parts[0]
            .text
            .clone()
            .unwrap_or_default())
    }
    fn get_tools_used(&self) -> Result<Vec<ToolCall>, AgentError> {
        Ok(self.candidates[0]
            .content
            .parts
            .iter()
            .filter_map(|part| part.function_call.clone())
            .map(|function_call| ToolCall {
                id: Some(function_call.name.clone()),
                call_type: Some("function".to_string()),
                function: FunctionCall {
                    name: function_call.name,
                    arguments: function_call.args,
                },
            })
            .collect())
    }
}

#[derive(Debug)]
pub struct GeminiServerModel {
    pub base_url: String,
    pub model_id: String,
    pub client: Client,
    pub temperature: f32,
    pub api_key: String,
    pub history: Option<Vec<Message>>,
}

impl GeminiServerModel {
    pub fn new(
        base_url: Option<&str>,
        model_id: Option<&str>,
        temperature: Option<f32>,
        api_key: Option<String>,
        history: Option<Vec<Message>>,
    ) -> Self {
        let api_key = api_key.unwrap_or_else(|| {
            std::env::var("GOOGLE_API_KEY").expect("GOOGLE_API_KEY must be set")
        });
        let model_id = model_id.unwrap_or("gemini-2.0-flash").to_string();
        let default_base_url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            model_id, api_key
        );
        let base_url = base_url.unwrap_or(default_base_url.as_str());
        let client = Client::new();
        GeminiServerModel {
            base_url: base_url.to_string(),
            model_id,
            client,
            temperature: temperature.unwrap_or(0.5),
            api_key,
            history,
        }
    }
}

pub struct GeminiServerModelBuilder {
    base_url: Option<String>,
    model_id: Option<String>,
    temperature: Option<f32>,
    api_key: Option<String>,
    history: Option<Vec<Message>>,
}

impl GeminiServerModelBuilder {
    pub fn new(model_id: &str) -> Self {
        Self {
            base_url: None,
            model_id: Some(model_id.to_string()),
            temperature: None,
            api_key: None,
            history: None,
        }
    }
    pub fn with_base_url(mut self, base_url: Option<&str>) -> Self {
        self.base_url = base_url.map(|s| s.to_string());
        self
    }
    pub fn with_model_id(mut self, model_id: Option<&str>) -> Self {
        self.model_id = model_id.map(|s| s.to_string());
        self
    }
    pub fn with_temperature(mut self, temperature: Option<f32>) -> Self {
        self.temperature = temperature;
        self
    }
    pub fn with_api_key(mut self, api_key: Option<&str>) -> Self {
        self.api_key = api_key.map(|s| s.to_string());
        self
    }
    pub fn with_history(mut self, history: Option<Vec<Message>>) -> Self {
        self.history = history;
        self
    }
    pub fn build(self) -> Result<GeminiServerModel> {
        Ok(GeminiServerModel::new(
            self.base_url.as_deref(),
            self.model_id.as_deref(),
            self.temperature,
            self.api_key,
            self.history,
        ))
    }
}

#[async_trait]
impl Model for GeminiServerModel {
    async fn run(
        &self,
        messages: Vec<Message>,
        history: Option<Vec<Message>>,
        tools_to_call_from: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
        let mut chat_contents = Vec::with_capacity(messages.len());

        if let Some(history) = history {
            for message in history {
                chat_contents.push(GeminiChatContent {
                    role: message.role.to_string(),
                    parts: vec![GeminiContentPart::Text(message.content)],
                });
            }
        }
        for message in messages {
            if !message.content.is_empty() {
                if message.role == MessageRole::System {
                    chat_contents.push(GeminiChatContent {
                        role: "user".to_string(),
                        parts: vec![GeminiContentPart::Text(message.content)],
                    });
                } else if message.role == MessageRole::Assistant {
                    chat_contents.push(GeminiChatContent {
                        role: "model".to_string(),
                        parts: vec![GeminiContentPart::Text(message.content)],
                    });
                } else {
                    chat_contents.push(GeminiChatContent {
                        role: message.role.to_string(),
                        parts: vec![GeminiContentPart::Text(message.content)],
                    });
                }
            }
        }

        let tools_to_call_from = if tools_to_call_from.is_empty() {
            None
        } else {
            Some(tools_to_call_from)
        };

        let stop_sequences = args.map(|args| args.get("stop").unwrap_or(&vec![]).to_vec());
        let request = GeminiChatRequest {
            contents: chat_contents,
            tools: tools_to_call_from.as_ref().map(|tools| GeminiTool {
                function_declarations: tools
                    .iter()
                    .map(|tool| {
                        let mut parameters = json!(tool.function.parameters.clone());
                        if let Value::Object(ref mut map) = parameters {
                            map.remove("$schema");
                            map.remove("title");
                            map.remove("additionalProperties");
                        }
                        json!({
                            "name": tool.function.name,
                            "description": tool.function.description,
                            "parameters": parameters
                        })
                    })
                    .collect(),
            }),
            generation_config: GeminiGenerationConfig {
                max_output_tokens: Some(max_tokens.unwrap_or(4500) as u32),
                temperature: Some(self.temperature),
                top_p: None,
                top_k: None,
                stop_sequences,
            },
        };

        let mut request = json!(request);
        if let Some(tools) = tools_to_call_from.as_ref() {
            request["tool_config"] = json!({
                "function_calling_config": { "mode": "ANY",
                "allowed_function_names": tools.iter().map(|tool| tool.function.name.to_string()).collect::<Vec<String>>() },

            });
        }
        println!(
            "Request: {}",
            serde_json::to_string_pretty(&request).unwrap()
        );

        let response = self
            .client
            .post(&self.base_url)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                AgentError::Generation(format!("Failed to get response from Gemini: {}", e))
            })?;
        match response.status() {
            reqwest::StatusCode::OK => {
                let response = response.json::<GeminiChatResponse>().await.unwrap();
                Ok(Box::new(response))
            }
            _ => Err(AgentError::Generation(format!(
                "Failed to get response from Gemini: {} {}",
                response.status(),
                response.text().await.unwrap(),
            ))),
        }
    }

    #[allow(unused_variables)]
    async fn run_stream(
        &self,
        messages: Vec<Message>,
        history: Option<Vec<Message>>,
        tools_to_call_from: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
        tx: broadcast::Sender<Status>,
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use crate::models::types::MessageRole;

    use super::*;

    #[tokio::test]
    async fn test_gemini_server_model() {
        let model = GeminiServerModelBuilder::new("gemini-2-flash")
            .build()
            .unwrap();
        let response = model
            .run(
                vec![Message {
                    role: MessageRole::User,
                    content: "Hello, how are you?".to_string(),
                    tool_call_id: None,
                    tool_calls: None,
                }],
                None,
                vec![],
                None,
                None,
            )
            .await
            .unwrap();

        println!("Response: {}", response.get_response().unwrap());
    }
}
