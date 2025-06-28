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
use futures::StreamExt;
use nanoid::nanoid;
use opentelemetry::{
    global,
    trace::{Span, Tracer},
    Context, KeyValue,
};
use reqwest::Client;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::broadcast;
use tokio::sync::mpsc::{channel, Receiver, Sender};

#[derive(Clone)]
pub enum Status {
    FirstContent(String),
    Content(String),
    ToolCallStart(String),
    ToolCallContent(String),
    Error(String),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OpenAIResponse {
    pub choices: Vec<Choice>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Choice {
    pub message: AssistantMessage,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AssistantMessage {
    pub role: MessageRole,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub refusal: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OpenAIStreamResponse {
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct StreamChoice {
    pub delta: DeltaMessage,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DeltaMessage {
    pub role: Option<MessageRole>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCallStream>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    #[serde(default = "generate_tool_id", deserialize_with = "deserialize_tool_id")]
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub call_type: Option<String>,
    pub function: FunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCallStream {
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub call_type: Option<String>,
    pub function: FunctionCallStream,
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCallStream {
    pub name: Option<String>,
    #[serde(serialize_with = "serialize_arguments")]
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

        let mut body = json!({
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        });

        let parent_cx = Context::current();
        let tracer = global::tracer("lumo");
        let mut span = tracer
            .span_builder("OpenAIServerModel::run")
            .with_start_time(std::time::SystemTime::now())
            .start_with_context(&tracer, &parent_cx);
        span.set_attributes(vec![
            KeyValue::new("input.value", serde_json::to_string(&messages).unwrap()),
            KeyValue::new("llm.model_name", self.model_id.clone()),
            KeyValue::new("gen_ai.request.temperature", self.temperature.to_string()),
            KeyValue::new("gen_ai.request.max_tokens", max_tokens.to_string()),
            KeyValue::new("timestamp", chrono::Utc::now().to_rfc3339()),
        ]);

        if let Some(args) = &args {
            for (key, value) in args {
                if key != "messages" && key != "tools" {
                    span.set_attributes(vec![
                        KeyValue::new(
                            format!("gen_ai.request.{}", key),
                            serde_json::to_string(value).unwrap(),
                        ),
                        KeyValue::new(
                            "gen_ai.tools",
                            serde_json::to_string(&tools_to_call_from).unwrap(),
                        ),
                    ]);
                }
            }
        }

        if !tools_to_call_from.is_empty() {
            body["tools"] = json!(tools_to_call_from);
            body["tool_choice"] = json!("required");
            span.set_attribute(KeyValue::new(
                "gen_ai.request.tool_choice",
                serde_json::to_string(&body["tool_choice"]).unwrap(),
            ));
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
                span.set_attribute(KeyValue::new(
                    "output.value",
                    serde_json::to_string_pretty(&response).unwrap(),
                ));
                span.end_with_timestamp(std::time::SystemTime::now());
                Ok(Box::new(response))
            }
            _ => Err(AgentError::Generation(format!(
                "Failed to get response from OpenAI: {} {}",
                response.status(),
                response.text().await.unwrap(),
            ))),
        }
    }

    async fn run_stream(
        &self,
        messages: Vec<Message>,
        history: Option<Vec<Message>>,
        tools_to_call_from: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
        tx: broadcast::Sender<Status>,
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
        let max_tokens = max_tokens.unwrap_or(4500);
        let mut messages = messages;
        if let Some(history) = history {
            messages = [history, messages].concat();
        }

        let mut body = json!({
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": true,
        });

        let parent_cx = Context::current();
        let tracer = global::tracer("lumo");
        let mut span = tracer
            .span_builder("OpenAIServerModel::run_stream")
            .with_start_time(std::time::SystemTime::now())
            .start_with_context(&tracer, &parent_cx);
        span.set_attributes(vec![
            KeyValue::new("input.value", serde_json::to_string(&messages).unwrap()),
            KeyValue::new("llm.model_name", self.model_id.clone()),
        ]);

        let parent_cx = Context::current();
        let tracer = global::tracer("lumo");
        let mut span = tracer
            .span_builder("OpenAIServerModel::run")
            .with_start_time(std::time::SystemTime::now())
            .start_with_context(&tracer, &parent_cx);
        span.set_attributes(vec![
            KeyValue::new("input.value", serde_json::to_string(&messages).unwrap()),
            KeyValue::new("llm.model_name", self.model_id.clone()),
            KeyValue::new("gen_ai.request.temperature", self.temperature.to_string()),
            KeyValue::new("gen_ai.request.max_tokens", max_tokens.to_string()),
            KeyValue::new("timestamp", chrono::Utc::now().to_rfc3339()),
        ]);

        if let Some(args) = &args {
            for (key, value) in args {
                if key != "messages" && key != "tools" {
                    span.set_attributes(vec![
                        KeyValue::new(
                            format!("gen_ai.request.{}", key),
                            serde_json::to_string(value).unwrap(),
                        ),
                        KeyValue::new(
                            "gen_ai.tools",
                            serde_json::to_string(&tools_to_call_from).unwrap(),
                        ),
                    ]);
                }
            }
        }

        if !tools_to_call_from.is_empty() {
            body["tools"] = json!(tools_to_call_from);
            body["tool_choice"] = json!("required");
            span.set_attribute(KeyValue::new(
                "gen_ai.request.tool_choice",
                serde_json::to_string(&body["tool_choice"]).unwrap(),
            ));
        }

        let stream = self
            .client
            .post(&self.base_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Accept", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("Connection", "keep-alive")
            .json(&body)
            .eventsource()
            .map_err(|e| AgentError::Generation(format!("Failed to create event source: {}", e)))?;

        let (tx_provider, rx_provider) = channel::<OpenAIStreamResponse>(32);
        tokio::spawn(forward_deserialized_chat_response_stream(
            stream,
            tx_provider,
        ));
        let response = process_stream_with_separate_tasks(rx_provider, tx)
            .await
            .map_err(|e| AgentError::Generation(format!("Failed to process stream: {}", e)))?;
        Ok(response)
    }
}

async fn forward_deserialized_chat_response_stream(
    mut stream: EventSource,
    tx: Sender<OpenAIStreamResponse>,
) -> anyhow::Result<()> {
    while let Some(event) = stream.next().await {
        let event = event?;
        if let Event::Message(event) = event {
            let data = serde_json::from_str::<OpenAIStreamResponse>(&event.data);
            match data {
                Ok(data) => {
                    tx.send(data).await?;
                }
                Err(e) => match event.data.contains("DONE") {
                    true => {
                        break;
                    }
                    false => {
                        eprintln!("Failed to deserialize response: {}", e);
                    }
                },
            }
        }
    }
    Ok(())
}

/// Demonstrates a better pattern for handling both accumulation and UI streaming
///
/// This pattern uses broadcast channels to efficiently handle multiple consumers
/// of the same stream data. It's more efficient than the current approach because:
/// 1. No blocking between accumulation and UI streaming
/// 2. No unnecessary cloning of content
/// 3. Better separation of concerns
/// 4. More scalable for multiple consumers
pub async fn process_stream_with_broadcast(
    mut stream: Receiver<OpenAIStreamResponse>,
    tx: broadcast::Sender<String>,
) -> Result<Box<dyn ModelResponse>, anyhow::Error> {
    let mut accumulated_content = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut current_tool_call: Option<ToolCall> = None;
    let mut current_arguments = String::new();

    // Process the original stream and broadcast
    while let Some(res) = stream.recv().await {
        if let Some(content) = &res.choices[0].delta.content {
            if let Err(e) = tx.send(content.clone()) {
                eprintln!("Failed to broadcast content: {}", e);
            }
            accumulated_content.push_str(content);
        }

        if let Some(tool_calls_delta) = &res.choices[0].delta.tool_calls {
            for tool_call_delta in tool_calls_delta {
                if let Some(id) = &tool_call_delta.id {
                    // New tool call starts, push the previous one if exists
                    if let Err(e) = tx.send(format!(
                        "Tool call started: {}",
                        tool_call_delta.function.name.clone().unwrap_or_default()
                    )) {
                        eprintln!("Failed to broadcast tool call content: {}", e);
                    }
                    if let Some(mut prev) = current_tool_call.take() {
                        prev.function.arguments = serde_json::from_str(&current_arguments).unwrap();
                        tool_calls.push(prev);
                        current_arguments = String::new();
                    }
                    current_tool_call = Some(ToolCall {
                        id: Some(id.clone()),
                        call_type: tool_call_delta.call_type.clone(),
                        function: FunctionCall {
                            name: tool_call_delta.function.name.clone().unwrap_or_default(),
                            arguments: Value::String(String::new()),
                        },
                    });
                }
                // Always update the current tool call's name and append arguments
                if let Some(current) = &mut current_tool_call {
                    if let Some(name) = &tool_call_delta.function.name {
                        current.function.name = name.clone();
                    }
                    let new_args = &tool_call_delta.function.arguments;
                    if let Value::String(new_str) = new_args {
                        if !new_str.is_empty() {
                            current_arguments.push_str(new_str);
                        }
                    } else {
                        current.function.arguments = new_args.clone();
                    }
                    // Broadcast tool call content for UI updates
                    let content_str = match &tool_call_delta.function.arguments {
                        Value::String(s) => s.clone(),
                        _ => serde_json::to_string(&tool_call_delta.function.arguments)
                            .unwrap_or_default(),
                    };
                    if let Err(e) = tx.send(content_str.clone()) {
                        eprintln!("Failed to broadcast tool call content: {}", e);
                    }
                }
            }
        }
    }
    // Push the last tool call if exists
    if let Some(mut last) = current_tool_call.take() {
        last.function.arguments = serde_json::from_str(&current_arguments).unwrap();
        tool_calls.push(last);
    }

    println!("Broadcast task completed");

    // Close the broadcast channel when stream ends
    drop(tx);

    // Create the final OpenAIResponse
    let response = Box::new(OpenAIResponse {
        choices: vec![Choice {
            message: AssistantMessage {
                role: MessageRole::Assistant,
                content: Some(accumulated_content),
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                refusal: None,
            },
        }],
    });

    Ok(response)
}

/// Alternative pattern that separates accumulation and broadcasting into different tasks
///
/// This pattern provides better error isolation and ensures that broadcasting
/// never blocks accumulation processing. Useful when you need:
/// 1. Heavy accumulation processing
/// 2. Multiple consumers with different needs
/// 3. Non-blocking UI updates
/// 4. Error isolation between accumulation and broadcasting
pub async fn process_stream_with_separate_tasks(
    mut stream: Receiver<OpenAIStreamResponse>,
    tx: broadcast::Sender<Status>,
) -> Result<Box<dyn ModelResponse>, anyhow::Error> {
    // Channel for communication between tasks
    let (accumulation_tx, mut accumulation_rx) = channel::<OpenAIStreamResponse>(32);

    let mut first_content = true;

    // Spawn accumulation task
    let accumulation_handle = tokio::spawn(async move {
        let mut accumulated_content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut current_tool_call: Option<ToolCall> = None;
        let mut current_arguments = String::new();

        while let Some(res) = accumulation_rx.recv().await {
            // Process content
            if let Some(content) = &res.choices[0].delta.content {
                accumulated_content.push_str(content);
            }

            // Process tool calls
            if let Some(tool_calls_delta) = &res.choices[0].delta.tool_calls {
                for tool_call_delta in tool_calls_delta {
                    if let Some(id) = &tool_call_delta.id {
                        // New tool call starts, push the previous one if exists
                        if let Some(mut prev) = current_tool_call.take() {
                            prev.function.arguments =
                                serde_json::from_str(&current_arguments).unwrap();
                            tool_calls.push(prev);
                            current_arguments = String::new();
                        }
                        current_tool_call = Some(ToolCall {
                            id: Some(id.clone()),
                            call_type: tool_call_delta.call_type.clone(),
                            function: FunctionCall {
                                name: tool_call_delta.function.name.clone().unwrap_or_default(),
                                arguments: Value::String(String::new()),
                            },
                        });
                    }

                    // Update current tool call
                    if let Some(current) = &mut current_tool_call {
                        if let Some(name) = &tool_call_delta.function.name {
                            current.function.name = name.clone();
                        }
                        let new_args = &tool_call_delta.function.arguments;
                        if let Value::String(new_str) = new_args {
                            if !new_str.is_empty() {
                                current_arguments.push_str(new_str);
                            }
                        } else {
                            current.function.arguments = new_args.clone();
                        }
                    }
                }
            }
        }
        // Push the last tool call if exists
        if let Some(mut last) = current_tool_call.take() {
            last.function.arguments = serde_json::from_str(&current_arguments).unwrap();
            tool_calls.push(last);
        }

        // Return accumulated data
        (accumulated_content, tool_calls)
    });

    // Spawn broadcasting task
    let tx_clone = tx.clone();
    let broadcast_handle = tokio::spawn(async move {
        while let Some(res) = stream.recv().await {
            // Forward to accumulation task
            if let Err(e) = accumulation_tx.send(res.clone()).await {
                eprintln!("Failed to send to accumulation task: {}", e);
                break;
            }

            // Broadcast content immediately
            if let Some(content) = &res.choices[0].delta.content {
                if first_content {
                    if let Err(e) = tx_clone.send(Status::FirstContent(content.clone())) {
                        eprintln!("Failed to broadcast first content: {}", e);
                    }
                    first_content = false;
                } else if let Err(e) = tx_clone.send(Status::Content(content.clone())) {
                    eprintln!("Failed to broadcast content: {}", e);
                }
            }

            // Broadcast tool call information
            // if let Some(tool_calls_delta) = &res.choices[0].delta.tool_calls {
            //     for tool_call_delta in tool_calls_delta {
            //         if let Some(id) = &tool_call_delta.id {
            //             if let Err(e) = tx_clone.send(format!("Tool call started: {}", tool_call_delta.function.name.clone().unwrap_or_default())) {
            //                 eprintln!("Failed to broadcast tool call start: {}", e);
            //             }
            //         }

            //         // Broadcast tool call content
            //         let content_str = match &tool_call_delta.function.arguments {
            //             Value::String(s) => s.clone(),
            //             _ => serde_json::to_string(&tool_call_delta.function.arguments).unwrap_or_default(),
            //         };
            //         if let Err(e) = tx_clone.send(content_str.clone()) {
            //             eprintln!("Failed to broadcast tool call content: {}", e);
            //         }
            //     }
            // }
        }

        // Close the accumulation channel
        drop(accumulation_tx);
    });

    // Wait for both tasks to complete
    let (accumulation_result, broadcast_result) =
        tokio::join!(accumulation_handle, broadcast_handle);

    // Handle any errors from the tasks
    let (accumulated_content, tool_calls) =
        accumulation_result.map_err(|e| anyhow::anyhow!("Accumulation task failed: {}", e))?;

    broadcast_result.map_err(|e| anyhow::anyhow!("Broadcast task failed: {}", e))?;

    // Close the broadcast channel when stream ends
    drop(tx);

    // Create the final OpenAIResponse
    let response = Box::new(OpenAIResponse {
        choices: vec![Choice {
            message: AssistantMessage {
                role: MessageRole::Assistant,
                content: Some(accumulated_content),
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                refusal: None,
            },
        }],
    });

    Ok(response)
}

#[cfg(test)]
mod tests {
    use crate::{
        models::types::MessageBuilder,
        prompts::CODE_SYSTEM_PROMPT,
        tools::{AnyTool, DuckDuckGoSearchTool},
    };

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

        let prompt = "What are patch embeddings?";
        let tool = DuckDuckGoSearchTool::new().tool_info();
        let response = model
            .run(
                vec![Message::new(MessageRole::User, prompt)],
                None,
                vec![tool],
                None,
                None,
            )
            .await
            .unwrap();
        let response = response.get_tools_used().unwrap();
        println!("{:?}", response);
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

    #[tokio::test]
    async fn test_openai_stream() -> Result<()> {
        let model = OpenAIServerModelBuilder::new("gpt-4o-mini")
            .with_base_url(Some("https://api.openai.com/v1/chat/completions"))
            .build()
            .unwrap();

        let prompt: &'static str = "What are patch embeddings?";

        let (tx, _rx) = broadcast::channel::<Status>(32);

        let stream = model
            .run_stream(
                vec![Message::new(MessageRole::User, prompt)],
                None,
                vec![],
                None,
                None,
                tx,
            )
            .await?;

        println!("\nFinal message: {}", stream.get_response().unwrap());
        Ok(())
    }

    #[tokio::test]
    async fn test_broadcast_pattern() -> Result<()> {
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
        while let Ok(status) = rx.recv().await {
            match status {
                Status::FirstContent(content) => {
                    ui_content.push_str(&content);
                    println!("First content: {}", content);
                }
                Status::Content(content) => {
                    ui_content.push_str(&content);
                    println!("Content: {}", content);
                }
                Status::ToolCallStart(tool_name) => {
                    println!("Tool call started: {}", tool_name);
                }
                Status::ToolCallContent(content) => {
                    println!("Tool call content: {}", content);
                }
                Status::Error(error) => {
                    eprintln!("Error: {}", error);
                }
            }
        }

        let accumulated_content = accumulated_response;
        println!(
            "Accumulated content: {:?}",
            accumulated_content.get_tools_used().unwrap()
        );
        println!("UI content: {}", ui_content);

        Ok(())
    }

    #[tokio::test]
    async fn test_separate_tasks_pattern() -> Result<()> {
        // This test demonstrates how you could use the separate tasks pattern
        // if you needed more complex processing or error isolation

        let model = OpenAIServerModelBuilder::new("gpt-4.1-mini")
            .with_base_url(Some("https://api.openai.com/v1/chat/completions"))
            .build()
            .unwrap();

        let prompt = "What are patch embeddings?";
        let tool = DuckDuckGoSearchTool::new().tool_info();

        let (tx, mut rx) = broadcast::channel::<Status>(32);

        // Create a mock stream for testing the separate tasks pattern
        let (mock_tx, mock_rx) = channel::<OpenAIStreamResponse>(32);

        // Spawn a task to simulate the stream processing with separate tasks
        let separate_tasks_handle =
            tokio::spawn(async move { process_stream_with_separate_tasks(mock_rx, tx).await });

        // Simulate some stream data
        tokio::spawn(async move {
            // This would normally come from the actual model stream
            // For testing, we'll just send a simple message
            let mock_response = OpenAIStreamResponse {
                choices: vec![StreamChoice {
                    delta: DeltaMessage {
                        role: Some(MessageRole::Assistant),
                        content: Some("Patch embeddings are...".to_string()),
                        tool_calls: None,
                    },
                }],
            };

            if let Err(e) = mock_tx.send(mock_response).await {
                eprintln!("Failed to send mock response: {}", e);
            }

            // Close the channel to end the stream
            drop(mock_tx);
        });

        // Process the broadcast stream
        let mut ui_content = String::new();
        while let Ok(status) = rx.recv().await {
            match status {
                Status::FirstContent(content) => {
                    ui_content.push_str(&content);
                    println!("First content: {}", content);
                }
                Status::Content(content) => {
                    ui_content.push_str(&content);
                    println!("Content: {}", content);
                }
                Status::ToolCallStart(tool_name) => {
                    println!("Tool call started: {}", tool_name);
                }
                Status::ToolCallContent(content) => {
                    println!("Tool call content: {}", content);
                }
                Status::Error(error) => {
                    eprintln!("Error: {}", error);
                }
            }
            println!("Separate tasks UI content: {}", ui_content);
        }

        // Get the final accumulated response
        let accumulated_response = separate_tasks_handle.await??;
        println!(
            "Separate tasks accumulated content: {}",
            accumulated_response.get_response().unwrap()
        );

        Ok(())
    }
}
