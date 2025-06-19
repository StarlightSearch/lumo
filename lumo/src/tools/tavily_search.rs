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
    #[schemars(description = "Optionally restrict results to a certain number of results")]
    max_results: Option<String>,
    #[schemars(description = "Optionally restrict results to a certain time range")]
    time_range: Option<String>,
    #[schemars(description = "Optionally restrict results to a certain number of days")]
    days: Option<String>,
    #[schemars(description = "Optionally restrict results to a certain number of results")]
    include_answer: Option<bool>,
    #[schemars(description = "Optionally restrict results to a certain number of results")]
    include_image_descriptions: Option<bool>,
    #[schemars(description = "Optionally restrict results to a certain number of results")]
    include_domains: Option<Vec<String>>,
    #[schemars(description = "Optionally restrict results to a certain number of results")]
    exclude_domains: Option<Vec<String>>,
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
fn default_topic() -> Option<Topic> {
    Some(Topic::General)
}

fn default_search_depth() -> Option<SearchDepth> {
    Some(SearchDepth::Basic)
}