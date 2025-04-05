use chrono;
use opentelemetry::{
    global::{self},
    trace::{SpanKind, TraceContextExt, Tracer},
    Context, KeyValue,
};
use serde_json::Value;
use tracing;

use crate::models::openai::ToolCall;

pub struct AgentTelemetry {
    tracer_name: String,
    current_context: Option<Context>,
}

impl AgentTelemetry {
    pub fn new(tracer_name: &str) -> Self {
        Self {
            tracer_name: tracer_name.to_string(),
            current_context: None,
        }
    }

    pub fn start_step(&mut self, step_number: i64) -> Context {
        let parent_cx = Context::current();
        let tracer_name = self.tracer_name.clone();
        let tracer = global::tracer(tracer_name);

        // Get current timestamp for consistent ordering
        let start_time = chrono::Utc::now().to_rfc3339();

        let span = tracer
            .span_builder(format!("Step {}", step_number))
            .with_kind(SpanKind::Internal)
            .with_start_time(std::time::SystemTime::now())
            .with_attributes(vec![
                KeyValue::new("gen_ai.operation.name", "agent_step"),
                KeyValue::new("step_type", "action"),
                KeyValue::new("step_number", step_number),
                KeyValue::new("timestamp", start_time),
            ])
            .start_with_context(&tracer, &parent_cx);

        let cx = Context::current_with_span(span);
        self.current_context = Some(cx.clone());
        cx
    }

    pub fn log_agent_memory(&self, agent_memory: &Value) {
        if let Some(cx) = &self.current_context {
            cx.span().set_attribute(KeyValue::new(
                "input.value",
                serde_json::to_string(agent_memory).unwrap_or_default(),
            ));
        }
    }

    pub fn log_tool_calls(&self, tools: &[ToolCall], cx: &Context) {
        cx.span()
            .set_attribute(KeyValue::new("gen_ai.tool_calls.count", tools.len() as i64));

        if !tools.is_empty() {
            cx.span().set_attribute(KeyValue::new(
                "gen_ai.tool_calls.names",
                tools
                    .iter()
                    .map(|t| t.function.name.clone())
                    .collect::<Vec<_>>()
                    .join(","),
            ));
        }

        tracing::info!(
            tool_calls = serde_json::to_string_pretty(tools).unwrap_or_default(),
            "Agent selected tools"
        );
    }

    pub fn log_tool_execution(
        &self,
        function_name: &str,
        arguments: &Value,
        cx: &Context,
    ) -> Context {
        let tracer = global::tracer("lumo");
        let span = tracer
            .span_builder(function_name.to_string())
            .with_kind(SpanKind::Internal)
            .with_attributes(vec![
                KeyValue::new("gen_ai.operation.name", "tool_calls"),
                KeyValue::new("timestamp", chrono::Utc::now().to_rfc3339()),
            ])
            .start_with_context(&tracer, &cx);
        let cx = Context::current_with_span(span);


        tracing::info!(
            tool = %function_name,
            args = ?arguments,
            "Executing tool call:"
        );
        cx.span()
            .set_attribute(KeyValue::new("gen_ai.tool.name", function_name.to_string()));
        cx.span().set_attributes(vec![
            KeyValue::new(
                "gen_ai.tool.arguments",
                serde_json::to_string(arguments).unwrap_or_default(),
            ),
            KeyValue::new(
                "input.value",
                serde_json::to_string(arguments).unwrap_or_default(),
            ),
        ]);
        cx
    }

    pub fn log_tool_result(&self, result: &str, success: bool, cx: &Context) {
        if success {
            cx.span()
                .set_attribute(KeyValue::new("gen_ai.tool.success", true));
        } else {
            cx.span()
                .set_attribute(KeyValue::new("gen_ai.tool.success", false));
            cx.span()
                .set_attribute(KeyValue::new("gen_ai.tool.error", result.to_string()));
            tracing::error!("Error executing tool call: {}", result);
        }
        cx.span()
            .set_attribute(KeyValue::new("output.value", result.to_string()));
    }

    pub fn log_final_answer(&self, answer: &str) {
        if let Some(cx) = &self.current_context {
            tracing::info!(answer = %answer, "Final answer received");
            cx.span()
                .set_attribute(KeyValue::new("output.value", answer.to_string()));
        }
    }

    pub fn log_observations(&self, observations: &[String]) {
        if let Some(cx) = &self.current_context {
            let observation_text = observations.join("\n");
            if observation_text.len() > 30000 {
                tracing::info!(
                    "Observation: {} \n ....This content has been truncated due to the 30000 character limit.....",
                    observation_text.chars().take(30000).collect::<String>()
                );
            } else {
                tracing::info!("Observation: {}", observation_text);
            }
            cx.span()
                .set_attribute(KeyValue::new("output.value", observation_text));
        }
    }

    pub fn end_step(&mut self) {
        if let Some(cx) = self.current_context.take() {
            // End the span with the current timestamp
            let current_time = chrono::Utc::now();
            cx.span().end_with_timestamp(current_time.into());

            // Small delay to allow the current span to be fully processed
            // before starting subsequent spans. This helps maintain proper
            // ordering when sending to remote telemetry backends.
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}
