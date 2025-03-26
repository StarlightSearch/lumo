use std::sync::Arc;

use lumo::agent::{Agent, FunctionCallingAgentBuilder};
use lumo::models::openai::OpenAIServerModelBuilder;
use lumo::tools::{AsyncTool, DuckDuckGoSearchTool, LanceRAGTool, PythonInterpreterTool, VisitWebsiteTool};
use embed_anything::embeddings::embed::EmbedderBuilder;

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
 

    let coding_agent = FunctionCallingAgentBuilder::new(model.clone())
        .with_tools(vec![Box::new(PythonInterpreterTool::new())])
        .with_name(Some("coding_agent"))
        .with_logging_level(Some(log::LevelFilter::Info))
        .with_description(Some("A coding agent that can write code in python. Use this to do calculations and other complex tasks."))
        .build()
        .unwrap();
    let dense_model = Arc::new(
        EmbedderBuilder::new()
            .model_architecture("jina")
            .model_id(Some("jinaai/jina-embeddings-v2-small-en"))
            .path_in_repo(Some("model.onnx"))
            .from_pretrained_onnx()
            .unwrap(),
    );
    let retrieval_tool = Box::new(LanceRAGTool::new(
        "C:\\Users\\arbal\\AppData\\Roaming\\com.starlight.starlight",
        "test",

        Box::new(move |text: String| Box::pin({
            let model = Arc::clone(&dense_model);
            async move {
                model.embed_query(&[text.as_str()], None)
                    .await
                    .unwrap()
                    .first()
                    .unwrap()
                    .embedding
                    .to_dense().unwrap()
            }
        })),
        10
    ).await.unwrap());
    
    let retrieval_agent = FunctionCallingAgentBuilder::new(model.clone())
    .with_tools(vec![retrieval_tool])
    .with_name(Some("retrieval_agent"))
    .with_logging_level(Some(log::LevelFilter::Info))
    .with_description(Some("A retrieval agent that can search for information in a database. Use this to find information about the user's query. This contains information related to deep learning, EEG analysis and other related topics."))
    .build()
    .unwrap();
    let mut agent = FunctionCallingAgentBuilder::new(model)
        .with_tools(tools)
        .with_max_steps(Some(10))
        .with_logging_level(Some(log::LevelFilter::Info))
        .with_planning_interval(Some(1))
        .with_managed_agents(vec![Box::new(coding_agent) as Box<dyn Agent>, Box::new(retrieval_agent) as Box<dyn Agent>])
        .build()
        .unwrap();

    let _result = agent.run("What are the stock prices of Apple and Nvidia. Get the ratio between the two.", true).await.unwrap();
 
}

