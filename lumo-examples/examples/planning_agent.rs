use futures::StreamExt;
use lumo::agent::{AgentStream, FunctionCallingAgentBuilder, Step};
use lumo::models::openai::OpenAIServerModelBuilder;
use lumo::tools::{AsyncTool, DuckDuckGoSearchTool, VisitWebsiteTool};

#[tokio::main]
async fn main() {
    let tools: Vec<Box<dyn AsyncTool>> = vec![
        Box::new(DuckDuckGoSearchTool::new()),
        Box::new(VisitWebsiteTool::new()),
    ];
    let model = OpenAIServerModelBuilder::new("gpt-4o-mini")
        .with_base_url(Some("https://api.openai.com/v1/chat/completions"))
        .build()
        .unwrap();

    let mut agent = FunctionCallingAgentBuilder::new(model)
        .with_tools(tools)
        .with_max_steps(Some(4))
        .build()
        .unwrap();
    // let _result = agent.run("What are the best restaurants in Eindhoven?", true).await.unwrap();
    let mut result = agent
        .stream_run("What are the best restaurants in Eindhoven?", true, None)
        .unwrap();

    while let Some(step) = result.next().await {
        match step {
            Ok(Step::PlanningStep(plan, facts)) => {
                println!("Plan: {}", plan);
                println!("Facts: {}", facts);
            }
            Ok(Step::ActionStep(action_step)) => {
                if let Some(final_answer) = action_step.final_answer {
                    println!("Final answer: {}", final_answer);
                }
                if let Some(tool_call) = action_step.tool_call {
                    println!("Tool call: {:?}", tool_call);
                }
            }

            _ => {}
        }
    }
}
