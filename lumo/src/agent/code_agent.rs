use anyhow::Result;
use async_trait::async_trait;
use opentelemetry::trace::{FutureExt, TraceContextExt};
use std::{collections::HashMap, mem::ManuallyDrop};
use tracing::{instrument, Span};

use crate::{
    errors::{AgentError, InterpreterError},
    local_python_interpreter::LocalPythonInterpreter,
    models::{
        model_traits::Model,
        openai::{FunctionCall, ToolCall},
        types::Message,
    },
    prompts::CODE_SYSTEM_PROMPT,
    telemetry::AgentTelemetry,
    tools::{AsyncTool, FinalAnswerTool},
};

use super::{agent_step::Step, agent_trait::Agent, multistep_agent::MultiStepAgent, AgentStep};

#[cfg(feature = "stream")]
use super::agent_trait::AgentStream;

#[cfg(feature = "code-agent")]
pub struct CodeAgent<M: Model> {
    base_agent: MultiStepAgent<M>,
    local_python_interpreter: ManuallyDrop<LocalPythonInterpreter>,
    telemetry: AgentTelemetry,
}

#[cfg(feature = "code-agent")]
impl<M: Model + Send + Sync + 'static> CodeAgent<M> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: Option<&str>,
        model: M,
        tools: Vec<Box<dyn AsyncTool>>,
        system_prompt: Option<&str>,
        managed_agents: Vec<Box<dyn Agent>>,
        description: Option<&str>,
        max_steps: Option<usize>,
        planning_interval: Option<usize>,
        history: Option<Vec<Message>>,
        logging_level: Option<log::LevelFilter>,
    ) -> Result<Self> {
        let system_prompt = system_prompt.unwrap_or(CODE_SYSTEM_PROMPT);

        let mut base_agent = MultiStepAgent::new(
            name,
            model,
            tools,
            Some(system_prompt),
            managed_agents,
            description,
            max_steps,
            planning_interval,
            history,
            logging_level,
        )?;

        let final_answer_tool = FinalAnswerTool::new();
        base_agent.tools.push(Box::new(final_answer_tool));

        let local_python_interpreter = LocalPythonInterpreter::new(Some(&base_agent.tools), None);

        Ok(Self {
            base_agent,
            local_python_interpreter: ManuallyDrop::new(local_python_interpreter),
            telemetry: AgentTelemetry::new("lumo"),
        })
    }
}

pub struct CodeAgentBuilder<'a, M: Model> {
    name: Option<&'a str>,
    model: M,
    tools: Vec<Box<dyn AsyncTool>>,
    system_prompt: Option<&'a str>,
    managed_agents: Vec<Box<dyn Agent>>,
    description: Option<&'a str>,
    max_steps: Option<usize>,
    planning_interval: Option<usize>,
    history: Option<Vec<Message>>,
    logging_level: Option<log::LevelFilter>,
}

impl<'a, M: Model + Send + Sync + 'static> CodeAgentBuilder<'a, M> {
    pub fn new(model: M) -> Self {
        Self {
            name: None,
            model,
            tools: vec![],
            system_prompt: None,
            managed_agents: vec![],
            description: None,
            max_steps: None,
            planning_interval: None,
            history: None,
            logging_level: None,
        }
    }
    pub fn with_name(mut self, name: Option<&'a str>) -> Self {
        self.name = name;
        self
    }
    pub fn with_tools(mut self, tools: Vec<Box<dyn AsyncTool>>) -> Self {
        self.tools = tools;
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
    pub fn with_logging_level(mut self, logging_level: Option<log::LevelFilter>) -> Self {
        self.logging_level = logging_level;
        self
    }
    pub fn build(self) -> Result<CodeAgent<M>> {
        CodeAgent::new(
            self.name,
            self.model,
            self.tools,
            self.system_prompt,
            self.managed_agents,
            self.description,
            self.max_steps,
            self.planning_interval,
            self.history,
            self.logging_level,
        )
    }
}

#[cfg(feature = "code-agent")]
#[async_trait]
impl<M: Model + Send + Sync + 'static> Agent for CodeAgent<M> {
    fn name(&self) -> &'static str {
        self.base_agent.name()
    }
    fn description(&self) -> &'static str {
        self.base_agent.description()
    }
    fn get_max_steps(&self) -> usize {
        self.base_agent.get_max_steps()
    }
    fn get_step_number(&self) -> usize {
        self.base_agent.get_step_number()
    }
    fn set_step_number(&mut self, step_number: usize) {
        self.base_agent.set_step_number(step_number)
    }
    fn increment_step_number(&mut self) {
        self.base_agent.increment_step_number()
    }
    fn get_logs_mut(&mut self) -> &mut Vec<Step> {
        self.base_agent.get_logs_mut()
    }
    fn reset_step_number(&mut self) {
        self.base_agent.reset_step_number()
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
    fn model(&self) -> &dyn Model {
        self.base_agent.model()
    }
    #[instrument(skip(self, log_entry), fields(step = ?self.get_step_number()))]
    async fn step(&mut self, log_entry: &mut Step) -> Result<Option<AgentStep>, AgentError> {
        let step_result = match log_entry {
            Step::ActionStep(step_log) => {
                let cx = self.telemetry.start_step(self.get_step_number() as i64);
                let span = Span::current();
                span.record("step_type", "action");
                let agent_memory = self.base_agent.write_inner_memory_from_logs(None)?;
                self.base_agent.input_messages = Some(agent_memory.clone());
                step_log.agent_memory = Some(agent_memory.clone());
                self.telemetry
                    .log_agent_memory(&serde_json::to_value(&agent_memory).unwrap_or_default());

                let llm_output = self
                    .base_agent
                    .model
                    .run(
                        self.base_agent.input_messages.as_ref().unwrap().clone(),
                        self.base_agent.history.clone(),
                        vec![],
                        None,
                        Some(HashMap::from([(
                            "stop".to_string(),
                            vec!["Observation:".to_string(), "<end_code>".to_string()],
                        )])),
                    )
                    .with_context(cx.clone())
                    .await?;

                let response = llm_output.get_response()?;
                step_log.llm_output = Some(response.clone());

                let code = match parse_code_blobs(&response) {
                    Ok(code) => code,
                    Err(e) => {
                        step_log.error = Some(e.clone());
                        tracing::info!("Error: {}", response + "\n" + &e.to_string());
                        self.telemetry.log_tool_result(&e.to_string(), false, &cx);
                        return Ok(Some(step_log.clone()));
                    }
                };

                tracing::info!("Code: {}", code);
                let tool_call = vec![ToolCall {
                    id: Some(format!("call_{}", nanoid::nanoid!())),
                    call_type: Some("function".to_string()),
                    function: FunctionCall {
                        name: "python_interpreter".to_string(),
                        arguments: serde_json::json!({ "code": code }),
                    },
                }];
                step_log.tool_call = Some(tool_call.clone());
                self.telemetry.log_tool_calls(&tool_call, &cx);

                tracing::info!(
                    tool_calls= serde_json::to_string_pretty(&step_log.tool_call.clone().unwrap()).unwrap_or_default(),
                    step = ?self.get_step_number(),
                    "Executing tool call:"
                );
                let result = self.local_python_interpreter.forward(&code);
                match result {
                    Ok(result) => {
                        let (result, execution_logs) = result;
                        let mut observation = match (execution_logs.is_empty(), result.is_empty()) {
                            (false, false) => {
                                format!("Execution logs: {}\nResult: {}", execution_logs, result)
                            }
                            (false, true) => format!("Execution logs: {}", execution_logs),
                            (true, false) => format!("Result: {}", result),
                            (true, true) => String::from("No output or logs generated"),
                        };
                        if observation.len() > 30000 {
                            observation = observation.chars().take(30000).collect::<String>();
                            observation = format!("{} \n....This content has been truncated due to the 30000 character limit.....", observation);
                        } else {
                            observation = observation.to_string();
                        }
                        tracing::info!("Observation: {}", observation);
                        self.telemetry.log_tool_result(&observation, true, &cx);
                        step_log.observations = Some(vec![observation]);
                    }
                    Err(e) => match e {
                        InterpreterError::FinalAnswer(answer) => {
                            step_log.final_answer = Some(answer.clone());
                            step_log.observations = Some(vec![format!("Final answer: {}", answer)]);
                            self.telemetry.log_final_answer(&answer);
                            cx.span().set_attribute(opentelemetry::KeyValue::new(
                                "end_time",
                                chrono::Utc::now().to_rfc3339(),
                            ));
                            cx.span().end_with_timestamp(std::time::SystemTime::now());
                            return Ok(Some(step_log.clone()));
                        }
                        _ => {
                            step_log.error = Some(AgentError::Execution(e.to_string()));
                            tracing::info!("Error: {}", e);
                            self.telemetry.log_tool_result(&e.to_string(), false, &cx);
                        }
                    },
                }
                self.telemetry
                    .log_observations(&step_log.observations.clone().unwrap_or_default());
                cx.span().set_attribute(opentelemetry::KeyValue::new(
                    "end_time",
                    chrono::Local::now().to_rfc3339(),
                ));
                cx.span().end_with_timestamp(std::time::SystemTime::now());
                step_log
            }
            _ => {
                todo!()
            }
        };

        Ok(Some(step_result.clone()))
    }
}

#[cfg(feature = "stream")]
impl<M: Model + std::fmt::Debug + Send + Sync + 'static> AgentStream for CodeAgent<M> {}

#[cfg(feature = "code-agent")]
pub fn parse_code_blobs(code_blob: &str) -> Result<String, AgentError> {
    use regex::Regex;

    let pattern = r"```(?:py|python)?\n([\s\S]*?)\n```";
    let re = Regex::new(pattern).map_err(|e| AgentError::Execution(e.to_string()))?;

    let matches: Vec<String> = re
        .captures_iter(code_blob)
        .map(|cap| cap[1].trim().to_string())
        .collect();

    if matches.is_empty() {
        // Check if it's a direct code blob or final answer
        if code_blob.contains("final") && code_blob.contains("answer") {
            return Err(AgentError::Parsing(
                "The code blob is invalid. It seems like you're trying to return the final answer. Use:\n\
                Code:\n\
                ```py\n\
                final_answer(\"YOUR FINAL ANSWER HERE\")\n\
                ```".to_string(),
            ));
        }

        return Err(AgentError::Parsing(
            "The code blob is invalid. Make sure to include code with the correct pattern, for instance:\n\
            Thoughts: Your thoughts\n\
            Code:\n\
            ```py\n\
            # Your python code here\n\
            ```".to_string(),
        ));
    }

    Ok(matches.join("\n\n"))
}
