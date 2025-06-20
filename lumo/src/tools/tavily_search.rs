//! This module contains the Tavily search tool.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::base::BaseTool;
use super::tool_traits::Tool;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "SearchDepth")]
pub enum SearchDepth {
    Basic,
    Advanced,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "Topic")]
pub enum Topic {
    General,
    News,
}

#[derive(Deserialize, Serialize, JsonSchema)]
#[schemars(title = "TavilySearchToolParams")]
pub struct TavilySearchToolParams {
    #[schemars(description = "The query to search for")]
    query: String,
    #[schemars(
        description = "Optionally restrict results to a certain topic",
        default = "default_topic"
    )]
    topic: Option<Topic>,
    #[schemars(
        description = "Optionally restrict results to a certain search depth",
        default = "default_search_depth"
    )]
    search_depth: Option<SearchDepth>,
    #[schemars(description = "Optionally restrict results to a certain number of results", 
    default = "default_max_results")]
    max_results: Option<String>,
    #[schemars(description = "Optionally include the answer")]
    #[serde(default = "default_true")]
    include_answer: Option<bool>,
    
    #[schemars(description = "Optionally include raw content")]
    #[serde(default = "default_true")]
    include_raw_content: Option<bool>,
    
    #[schemars(description = "Optionally include images")]
    #[serde(default = "default_false")]
    include_images: Option<bool>,
    
    #[schemars(description = "Optionally include image descriptions")]
    #[serde(default = "default_false")]
    include_image_descriptions: Option<bool>,
    
    #[schemars(description = "Optionally restrict results to a certain number of domains")]
    #[serde(default = "default_vec_string")]
    include_domains: Option<Vec<String>>,
    
    #[schemars(description = "Optionally restrict results to a certain number of excluded domains")]
    #[serde(default = "default_vec_string")]
    exclude_domains: Option<Vec<String>>,
    
    #[schemars(description = "Optionally restrict results to a certain country")]
    #[serde(default = "default_country")]
    country: Option<String>,
}



#[derive(Debug, Serialize, Default, Clone)]
pub struct TavilySearchTool {
    pub tool: BaseTool,
    pub api_key: String,
}

impl TavilySearchTool {
    pub fn new(api_key: Option<String>) -> Self {
        let tool = BaseTool {
            name: "tavily_search",
            description: "Tavily Search Tool",
        };
        Self {
            tool,
            api_key: api_key.unwrap_or_default(),
        }
    }

    pub async fn forward(&self, arguments: TavilySearchToolParams) -> Result<String> {
        let client = reqwest::Client::new();
        let response = client
            .post("https://api.tavily.com")
            .json(&arguments)
            .header("Bearer", &self.api_key)
            .header("Content-Type", "application/json")
            .send()
            .await;

        match response {
            Ok(resp) => {
                if resp.status().is_success() {
                    let results: serde_json::Value = resp.json().await?;
                    Ok(results.to_string())
                } else {
                    Err(anyhow!(
                        "Failed to fetch search results: HTTP {}, Error: {}",
                        resp.status(),
                        resp.text().await.unwrap()
                    ))
                }
            }
            Err(e) => Err(anyhow!("Failed to make the request: {}", e)),
        }
    }
}

#[async_trait]
impl Tool for TavilySearchTool {
    type Params = TavilySearchToolParams;
    fn name(&self) -> &'static str {
        self.tool.name
    }
    fn description(&self) -> &'static str {
        self.tool.description
    }

    async fn forward(&self, arguments: TavilySearchToolParams) -> Result<String> {
        self.forward(arguments).await
    }
}


// Default implementations
fn default_topic() -> Option<Topic> {Some(Topic::General)}
fn default_search_depth() -> Option<SearchDepth> {Some(SearchDepth::Basic)}
fn default_max_results() -> Option<String> {Some("10".to_string())}
fn default_true() -> Option<bool> { Some(true) }
fn default_false() -> Option<bool> { Some(false) }
fn default_vec_string() -> Option<Vec<String>> { Some(vec![]) }
fn default_country() -> Option<String> { None }

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_tavily_search_tool() {
        let tool = TavilySearchTool::new(None);
        let _result = tool.forward(TavilySearchToolParams {
            query: "What is lumo?".to_string(),
            topic: None,
            search_depth: None,
            max_results: None,
            include_answer: None,
            include_raw_content: None,
            include_images: None,
            include_image_descriptions: None,
            include_domains: None,
            exclude_domains: None,
            country: None,
        }).await;
        println!("{:?}", _result);
    }
}