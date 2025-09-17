use std::collections::HashMap;

use crate::{
    agent::parse_response,
    errors::AgentError,
    models::{
        model_traits::Model,
        openai::{FunctionCall, ToolCall},
        types::Message,
    },
    prompts::TOOL_CALLING_SYSTEM_PROMPT,
    telemetry::AgentTelemetry,
    tools::{ToolFunctionInfo, ToolGroup, ToolInfo, ToolType},
};
use anyhow::Result;
use async_trait::async_trait;
use futures::future::join_all;
use mcp_client::{McpClient, McpClientTrait, TransportHandle};
use mcp_core::{Content, Tool};
use opentelemetry::trace::{FutureExt, TraceContextExt};
use serde_json::json;
use tracing::instrument;

use super::{Agent, AgentStep, MultiStepAgent, Step};

#[cfg(feature = "stream")]
use super::agent_trait::AgentStream;

fn initialize_system_prompt(system_prompt: String, tools: Vec<Tool>) -> Result<String> {
    let tool_names = tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect::<Vec<_>>();
    let tool_description = serde_json::to_string(&tools)?;
    let mut system_prompt = system_prompt.replace("{{tool_names}}", &tool_names.join(", "));
    system_prompt = system_prompt.replace("{{tool_descriptions}}", &tool_description);
    system_prompt = system_prompt.replace("{{current_time}}", &chrono::Local::now().to_string());
    Ok(system_prompt)
}

pub struct McpAgent<M, S>
where
    M: Model + Send + Sync + 'static,
    S: TransportHandle + 'static,
{
    base_agent: MultiStepAgent<M>,
    mcp_clients: Vec<McpClient<S>>,
    tools: Vec<Tool>,
    telemetry: AgentTelemetry,  
}

impl From<Tool> for ToolInfo {
    fn from(tool: Tool) -> Self {
        let schema =
            serde_json::from_value::<serde_json::Value>(tool.input_schema).unwrap_or_default();
        ToolInfo {
            tool_type: ToolType::Function,
            function: ToolFunctionInfo {
                name: tool.name,
                description: tool.description,
                parameters: schema,
            },
        }
    }
}

impl From<ToolInfo> for Tool {
    fn from(tool: ToolInfo) -> Self {
        Tool::new(
            tool.function.name,
            tool.function.description,
            serde_json::to_value(tool.function.parameters).unwrap(),
            None,
        )
    }
}

impl<M, S> McpAgent<M, S>
where
    M: Model + std::fmt::Debug + Send + Sync,
    S: TransportHandle + 'static,
{
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        name: Option<&str>,
        model: M,
        system_prompt: Option<&str>,
        managed_agents: Vec<Box<dyn Agent>>,
        description: Option<&str>,
        max_steps: Option<usize>,
        mcp_clients: Vec<McpClient<S>>,
        planning_interval: Option<usize>,
        history: Option<Vec<Message>>,
        logging_level: Option<log::LevelFilter>,
    ) -> Result<Self> {
        let system_prompt = match system_prompt {
            Some(prompt) => prompt.to_string(),
            None => TOOL_CALLING_SYSTEM_PROMPT.to_string(),
        };
        let mut tools = Vec::new();
        for client in &mcp_clients {
            tools.extend(client.list_tools(None, None).await?.tools);
        }
        let description = match description {
            Some(desc) => desc.to_string(),
            None => "A multi-step agent that can solve tasks using a series of tools".to_string(),
        };
        let base_agent = MultiStepAgent::new(
            name,
            model,
            vec![],
            Some(&initialize_system_prompt(system_prompt, tools.clone())?),
            managed_agents,
            Some(&description),
            max_steps,
            planning_interval,
            history,
            logging_level,
        )?;
        Ok(Self {
            base_agent,
            mcp_clients,
            tools: tools.to_vec(),
            telemetry: AgentTelemetry::new("lumo"),
        })
    }
}

pub struct McpAgentBuilder<'a, M, S>
where
    M: Model + std::fmt::Debug + Send + Sync,
    S: TransportHandle + 'static,
{
    name: Option<&'a str>,
    model: M,
    system_prompt: Option<&'a str>,
    managed_agents: Vec<Box<dyn Agent>>,
    description: Option<&'a str>,
    max_steps: Option<usize>,
    planning_interval: Option<usize>,
    history: Option<Vec<Message>>,
    mcp_clients: Vec<McpClient<S>>,
    logging_level: Option<log::LevelFilter>,
}

impl<'a, M, S> McpAgentBuilder<'a, M, S>
where
    M: Model + std::fmt::Debug + Send + Sync,
    S: TransportHandle + 'static,
{
    pub fn new(model: M) -> Self {
        Self {
            name: None,
            model,
            system_prompt: None,
            managed_agents: vec![],
            description: None,
            max_steps: None,
            planning_interval: None,
            history: None,
            mcp_clients: vec![],
            logging_level: None,
        }
    }
    pub fn with_name(mut self, name: Option<&'a str>) -> Self {
        self.name = name;
        self
    }
    pub fn with_system_prompt(mut self, system_prompt: Option<&'a str>) -> Self {
        self.system_prompt = system_prompt;
        self
    }
    pub fn with_managed_agents(mut self, managed_agents: Vec<Box<dyn Agent>>) -> Self {
        self.managed_agents = managed_agents;
        self
    }
    pub fn with_description(mut self, description: Option<&'a str>) -> Self {
        self.description = description;
        self
    }
    pub fn with_max_steps(mut self, max_steps: Option<usize>) -> Self {
        self.max_steps = max_steps;
        self
    }
    pub fn with_planning_interval(mut self, planning_interval: Option<usize>) -> Self {
        self.planning_interval = planning_interval;
        self
    }
    pub fn with_history(mut self, history: Option<Vec<Message>>) -> Self {
        self.history = history;
        self
    }
    pub fn with_mcp_clients(mut self, mcp_clients: Vec<McpClient<S>>) -> Self {
        self.mcp_clients = mcp_clients;
        self
    }
    pub fn with_logging_level(mut self, logging_level: Option<log::LevelFilter>) -> Self {
        self.logging_level = logging_level;
        self
    }
    pub async fn build(self) -> Result<McpAgent<M, S>> {
        McpAgent::new(
            self.name,
            self.model,
            self.system_prompt,
            self.managed_agents,
            self.description,
            self.max_steps,
            self.mcp_clients,
            self.planning_interval,
            self.history,
            self.logging_level,
        )
        .await
    }
}

#[async_trait]
impl<M, S> Agent for McpAgent<M, S>
where
    M: Model + std::fmt::Debug + Send + Sync,
    S: TransportHandle + 'static,
{
    fn name(&self) -> &'static str {
        self.base_agent.name()
    }
    fn description(&self) -> &'static str {
        self.base_agent.description()
    }
    fn set_task(&mut self, task: &str) {
        self.base_agent.set_task(task);
    }
    fn get_task(&self) -> &str {
        self.base_agent.get_task()
    }
    fn get_system_prompt(&self) -> &str {
        self.base_agent.get_system_prompt()
    }
    fn get_max_steps(&self) -> usize {
        self.base_agent.get_max_steps()
    }
    fn get_step_number(&self) -> usize {
        self.base_agent.get_step_number()
    }
    fn reset_step_number(&mut self) {
        self.base_agent.reset_step_number();
    }
    fn set_step_number(&mut self, step_number: usize) {
        self.base_agent.set_step_number(step_number)
    }
    fn increment_step_number(&mut self) {
        self.base_agent.increment_step_number();
    }
    fn get_logs_mut(&mut self) -> &mut Vec<Step> {
        self.base_agent.get_logs_mut()
    }
    fn model(&self) -> &dyn Model {
        self.base_agent.model()
    }
    fn get_planning_interval(&self) -> Option<usize> {
        self.base_agent.get_planning_interval()
    }
    fn set_planning_interval(&mut self, planning_interval: Option<usize>) {
        self.base_agent.set_planning_interval(planning_interval);
    }
    async fn planning_step(
        &mut self,
        task: &str,
        is_first_step: bool,
        step: usize,
    ) -> Result<Option<Step>> {
        self.base_agent
            .planning_step(task, is_first_step, step)
            .await
    }
    /// Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
    ///
    /// Returns None if the step is not final.
    #[instrument(skip(self, log_entry), fields(step = ?self.get_step_number()))]
    async fn step(&mut self, log_entry: &mut Step) -> Result<Option<AgentStep>, AgentError> {
        match log_entry {
            Step::ActionStep(step_log) => {
                let cx = self.telemetry.start_step(self.get_step_number() as i64);

                let agent_memory = self.base_agent.write_inner_memory_from_logs(None)?;
                self.base_agent.input_messages = Some(agent_memory.clone());
                step_log.agent_memory = Some(agent_memory.clone());
                self.telemetry
                    .log_agent_memory(&serde_json::to_value(&agent_memory).unwrap_or_default());
                let mut tools = self
                    .tools
                    .iter()
                    .cloned()
                    .map(ToolInfo::from)
                    .collect::<Vec<_>>();

                let managed_agents = self
                    .base_agent
                    .managed_agents
                    .iter()
                    .map(|agent| ToolInfo {
                        tool_type: ToolType::Function,
                        function: ToolFunctionInfo {
                            name: agent.name().to_string(),
                            description: agent.description().to_string(),
                            parameters: json!({
                                "type": "object",
                                "properties": {
                                    "task": {
                                        "type": "string",
                                        "description": "The task to perform"
                                    }
                                }
                            }),
                        },
                    })
                    .collect::<Vec<_>>();

                tools.extend(managed_agents);

                // Add final answer tool
                // let final_answer_tool = ToolInfo::from(Tool::new(
                //     "final_answer",
                //     "Use this to provide your final answer to the user's request",
                //     serde_json::json!({
                //         "type": "object",
                //         "properties": {
                //             "answer": {
                //                 "type": "string",
                //                 "description": "The final answer to provide to the user"
                //             }
                //         },
                //         "required": ["answer"]
                //     }),
                // ));
                // tools.push(final_answer_tool);

                tracing::debug!("Starting model inference with {} tools", tools.len());
                let model_message = self
                    .base_agent
                    .model
                    .run(
                        self.base_agent.input_messages.as_ref().unwrap().clone(),
                        self.base_agent.history.clone(),
                        tools,
                        None,
                        Some(HashMap::from([(
                            "stop".to_string(),
                            vec!["Observation:".to_string()],
                        )])),
                    )
                    .with_context(cx.clone())
                    .await?;

                step_log.llm_output = Some(model_message.get_response().unwrap_or_default());
                let mut observations = Vec::new();
                let mut tools = model_message.get_tools_used()?;

                step_log.tool_call = if tools.is_empty() {
                    None
                } else {
                    Some(tools.clone())
                };

                self.telemetry.log_tool_calls(&tools, &cx);

                if let Ok(response) = model_message.get_response() {
                    if !response.trim().is_empty() {
                        if let Ok(action) = parse_response(&response) {
                            tools = vec![ToolCall {
                                id: Some(format!("call_{}", nanoid::nanoid!())),
                                call_type: Some("function".to_string()),
                                function: FunctionCall {
                                    name: action["name"].as_str().unwrap_or_default().to_string(),
                                    arguments: action["arguments"].clone(),
                                },
                            }];

                            step_log.tool_call = Some(tools.clone());
                            self.telemetry.log_tool_calls(&tools, &cx);
                        }
                    }
                    if tools.is_empty() {
                        self.base_agent.write_inner_memory_from_logs(None)?;
                        step_log.final_answer = Some(response.clone());
                        step_log.observations = Some(vec![response.clone()]);
                        self.telemetry.log_final_answer(&response);
                        cx.span().end_with_timestamp(std::time::SystemTime::now());
                        return Ok(Some(step_log.clone()));
                    }
                }

                let managed_agent_names = self
                    .base_agent
                    .managed_agents
                    .iter()
                    .map(|agent| agent.name())
                    .collect::<Vec<_>>();

                let mut called_tools = Vec::new();
                for tool in &tools {
                    let function_name = tool.clone().function.name;

                    match function_name.as_str() {
                        "final_answer" => {
                            tracing::info!(answer = ?tool.function.arguments, "Final answer received");
                            let answer = self.base_agent.tools.call(&tool.function).await?;
                            step_log.observations = Some(vec![answer.clone()]);
                            step_log.final_answer = Some(answer.clone());
                            return Ok(Some(step_log.clone()));
                        }
                        _ => {
                            tracing::info!(
                                tool = %function_name,
                                args = ?tool.function.arguments,
                                "Executing tool call:"
                            );
                            called_tools.push(tool.function.clone());

                            let mut futures = Vec::new();

                            if !managed_agent_names.contains(&function_name.as_str()) {
                                // Run tool
                                {
                                    for client in &self.mcp_clients {
                                        if client
                                            .list_tools(None, None)
                                            .await
                                            .map_err(|e| AgentError::Execution(e.to_string()))?
                                            .tools
                                            .iter()
                                            .any(|t| t.name == tool.function.name)
                                        {
                                            futures.push(client.call_tool(
                                                &tool.function.name,
                                                tool.function.arguments.clone(),
                                                None,
                                            ));
                                        }
                                    }
                                }
                            } else {
                                // Run managed agent
                                let task = tool.function.arguments.get("task");
                                if let Some(task) = task {
                                    if let Some(task_str) = task.as_str() {
                                        tracing::info!(
                                            tool = %function_name,
                                            args = ?tool.function.arguments,
                                            "Executing tool call: Agent Selected {}",
                                            function_name
                                        );
                                        let result = self
                                            .base_agent
                                            .managed_agents
                                            .iter_mut()
                                            .find(|agent| agent.name() == function_name.as_str())
                                            .unwrap()
                                            .run(task_str, true)
                                            .await?;
                                        observations.push(result);
                                    }
                                }
                            }
                            let results = join_all(futures).await;
                            for (i, result) in results.into_iter().enumerate() {
                                let cx = self.telemetry.log_tool_execution(
                                    &called_tools[i].name,
                                    &called_tools[i].arguments,
                                    &cx,
                                );
                                match result {
                                    Ok(observation) => {
                                        let text = observation
                                            .content
                                            .iter()
                                            .map(|content| match content {
                                                Content::Text(text) => text.text.clone(),
                                                _ => "".to_string(),
                                            })
                                            .collect::<Vec<_>>()
                                            .join("\n");
                                        let formatted = format!(
                                            "Observation from {}: {}",
                                            function_name,
                                            text.chars().take(30000).collect::<String>()
                                        );
                                        tracing::debug!(
                                            tool = %function_name,
                                            observation = %formatted,
                                            "Tool call succeeded"
                                        );
                                        self.telemetry.log_tool_result(&text, true, &cx);

                                        observations.push(formatted);
                                    }
                                    Err(e) => {
                                        let error_msg =
                                            format!("Error from {}: {}", function_name, e);
                                        tracing::error!(
                                            tool = %function_name,
                                            error = %e,
                                            "Tool call failed"
                                        );
                                        self.telemetry.log_tool_result(&error_msg, false, &cx);

                                        observations.push(error_msg);
                                    }
                                }
                                cx.span().end_with_timestamp(std::time::SystemTime::now());
                            }
                        }
                    }
                }
                step_log.observations = Some(observations);

                if step_log
                    .observations
                    .clone()
                    .unwrap_or_default()
                    .join("\n")
                    .trim()
                    .len()
                    > 30000
                {
                    tracing::debug!(
                        "Observation: {} \n ....This content has been truncated due to the 30000 character limit.....",
                        step_log.observations.clone().unwrap_or_default().join("\n").trim().chars().take(30000).collect::<String>()
                    );
                } else {
                    tracing::debug!(
                        "Observation: {}",
                        step_log.observations.clone().unwrap_or_default().join("\n")
                    );
                }
                cx.span().end_with_timestamp(std::time::SystemTime::now());
                Ok(Some(step_log.clone()))
            }
            _ => {
                todo!()
            }
        }
    }
}

#[cfg(feature = "stream")]
impl<M, S> AgentStream for McpAgent<M, S>
where
    M: Model + std::fmt::Debug + Send + Sync,
    S: TransportHandle + 'static,
{
}
