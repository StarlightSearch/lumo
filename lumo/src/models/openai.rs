use std::collections::HashMap;

use crate::{
    errors::AgentError,
    models::{
        model_traits::{Model, ModelResponse},
        types::{Message, MessageRole},
    },
    tools::tool_traits::ToolInfo,
};
use anyhow::Result;
use async_trait::async_trait;
use nanoid::nanoid;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Deserialize)]
pub struct OpenAIResponse {
    pub choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: AssistantMessage,
}

#[derive(Debug, Deserialize)]
pub struct AssistantMessage {
    pub role: MessageRole,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub refusal: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    #[serde(default = "generate_tool_id", deserialize_with = "deserialize_tool_id")]
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub call_type: Option<String>,
    pub function: FunctionCall,
}

fn generate_tool_id() -> Option<String> {
    Some(nanoid!(16))
}

fn deserialize_tool_id<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt_id = Option::<String>::deserialize(deserializer)?;
    if let Some(id) = opt_id {
        if id.is_empty() {
            Ok(Some(nanoid!(16)))
        } else {
            Ok(Some(id))
        }
    } else {
        Ok(Some(nanoid!(16)))
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    #[serde(
        deserialize_with = "deserialize_arguments",
        serialize_with = "serialize_arguments"
    )]
    pub arguments: Value,
}

// Update the serialize_arguments function to handle JSON objects properly
fn serialize_arguments<S>(value: &Value, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    // If it's already a string, return as is
    if let Value::String(s) = value {
        serializer.serialize_str(s)
    } else {
        // For objects and other types, serialize to a JSON string
        match serde_json::to_string(value) {
            Ok(json_str) => serializer.serialize_str(&json_str),
            Err(e) => Err(serde::ser::Error::custom(format!(
                "JSON serialization error: {}",
                e
            ))),
        }
    }
}

// Update the deserialize_arguments function to be more robust
fn deserialize_arguments<'de, D>(deserializer: D) -> Result<Value, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;

    match &value {
        Value::String(s) => {
            // Try to parse the string as JSON
            match serde_json::from_str(s) {
                Ok(parsed) => Ok(parsed),
                Err(_) => Ok(value), // If parsing fails, return the original string value
            }
        }
        _ => Ok(value), // If it's not a string, return as is
    }
}

impl ModelResponse for OpenAIResponse {
    fn get_response(&self) -> Result<String, AgentError> {
        Ok(self
            .choices
            .first()
            .ok_or(AgentError::Generation(
                "No message returned from OpenAI".to_string(),
            ))?
            .message
            .content
            .clone()
            .unwrap_or_default())
    }

    fn get_tools_used(&self) -> Result<Vec<ToolCall>, AgentError> {
        Ok(self
            .choices
            .first()
            .ok_or(AgentError::Generation(
                "No message returned from OpenAI".to_string(),
            ))?
            .message
            .tool_calls
            .clone()
            .unwrap_or_default())
    }
}

#[derive(Debug, Clone)]
pub struct OpenAIServerModel {
    pub base_url: String,
    pub model_id: String,
    pub client: Client,
    pub temperature: f32,
    pub api_key: String,
    pub history: Option<Vec<Message>>,
}

impl OpenAIServerModel {
    pub fn new(
        base_url: Option<&str>,
        model_id: Option<&str>,
        temperature: Option<f32>,
        api_key: Option<String>,
        history: Option<Vec<Message>>,
    ) -> Self {
        let api_key = api_key.unwrap_or_else(|| {
            std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set")
        });
        let model_id = model_id.unwrap_or("gpt-4o-mini").to_string();
        let base_url = base_url.unwrap_or("https://api.openai.com/v1/chat/completions");
        let client = Client::new();
        OpenAIServerModel {
            base_url: base_url.to_string(),
            model_id,
            client,
            temperature: temperature.unwrap_or(0.5),
            api_key,
            history,
        }
    }
}

pub struct OpenAIServerModelBuilder {
    base_url: Option<String>,
    model_id: Option<String>,
    temperature: Option<f32>,
    api_key: Option<String>,
    history: Option<Vec<Message>>,
}

impl OpenAIServerModelBuilder {
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
    pub fn build(self) -> Result<OpenAIServerModel> {
        Ok(OpenAIServerModel::new(
            self.base_url.as_deref(),
            self.model_id.as_deref(),
            self.temperature,
            self.api_key,
            self.history,
        ))
    }
}

#[async_trait]
impl Model for OpenAIServerModel {
    async fn run(
        &self,
        messages: Vec<Message>,
        history: Option<Vec<Message>>,
        tools_to_call_from: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
        let max_tokens = max_tokens.unwrap_or(4500);
        let mut messages = messages;
        if let Some(history) = history {
            messages = [history, messages].concat();
        }
        // let messages = messages
        //     .iter()
        //     .map(|message| {
        //         json!({
        //             "role": message.role,
        //             "content": message.content,
        //         })
        //     })
        //     .collect::<Vec<Value>>();
        let mut body = json!({
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        });

        if !tools_to_call_from.is_empty() {
            body["tools"] = json!(tools_to_call_from);
            body["tool_choice"] = json!("auto");
        }

        if let Some(args) = args {
            let body_map = body.as_object_mut().unwrap();
            for (key, value) in args {
                body_map.insert(key, json!(value));
            }
        }


        let response = self
            .client
            .post(&self.base_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                AgentError::Generation(format!("Failed to get response from OpenAI: {}", e))
            })?;

        match response.status() {
            reqwest::StatusCode::OK => {
                let response = response.json::<OpenAIResponse>().await.unwrap();
                Ok(Box::new(response))
            }
            _ => Err(AgentError::Generation(format!(
                "Failed to get response from OpenAI: {} {}",
                response.status(),
                response.text().await.unwrap(),
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{models::types::MessageBuilder, prompts::CODE_SYSTEM_PROMPT};

    use super::*;

    fn create_tool_response_message_from_tool_call() -> Vec<Message> {
        vec![
            MessageBuilder::new(MessageRole::System, CODE_SYSTEM_PROMPT).build(),
            MessageBuilder::new(MessageRole::User, "Who is elon musk?").build(),
            MessageBuilder::new(MessageRole::Assistant, "tool request").with_tool_calls(vec![serde_json::from_value::<ToolCall>(json!({

                        "function": {
                    "arguments": "{\"code\":\"elon_musk_info = duckduckgo_search(query=\\\"who is Elon Musk\\\")\\nprint(elon_musk_info)\"}",
                    "name": "python_interpreter"
                  },
                  "id": "call_RrkJwA8V1BbYJNLvZH37e",
                  "type": "function"
                    
            })).unwrap()]).build(),
            MessageBuilder::new(MessageRole::ToolResponse, "This is a tool response").with_tool_call_id("call_RrkJwA8V1BbYJNLvZH37e").build(),
        ]
    }

    #[tokio::test]
    async fn test_openai_with_tool_response() {
        let model = OpenAIServerModelBuilder::new("gpt-4o-mini")
            .with_base_url(Some("https://api.openai.com/v1/chat/completions"))
            .build()
            .unwrap();
        let response = model
            .run(
                create_tool_response_message_from_tool_call(),
                None,
                vec![],
                None,
                None,
            )
            .await
            .unwrap();
        let response = response.get_response().unwrap();
        println!("{}", response);
    }

    #[tokio::test]
    async fn test_gemini_with_tool_response() {
        let model = OpenAIServerModelBuilder::new("gemini-2.0-flash")
            .with_base_url(Some(
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            ))
            .with_api_key(Some(&std::env::var("GEMINI_API_KEY").unwrap()))
            .build()
            .unwrap();
        let response = model
            .run(
                create_tool_response_message_from_tool_call(),
                None,
                vec![],
                None,
                None,
            )
            .await
            .unwrap();
        let response = response.get_response().unwrap();
        println!("{}", response);
    }
}
