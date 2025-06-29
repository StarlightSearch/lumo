//! This module contains the Tavily search tool.

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::base::BaseTool;
use super::tool_traits::Tool;
use anyhow::Result;

#[derive(Deserialize, JsonSchema)]
#[schemars(title = "TavilySearchToolParams")]
pub struct TavilySearchToolParams {
    #[schemars(description = "The query to search for")]
    query: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct TavilySearchResponse {
    pub results: Vec<TavilySearchResult>,
}

#[derive(Debug, Deserialize, Default)]
pub struct TavilySearchResult {
    #[serde(default, deserialize_with = "deserialize_null_as_empty_string")]
    pub title: String,
    #[serde(default, deserialize_with = "deserialize_null_as_empty_string")]
    pub url: String,
    #[serde(default, deserialize_with = "deserialize_null_as_empty_string")]
    pub content: String,
    #[serde(default)]
    pub score: f64,
    #[serde(default, deserialize_with = "deserialize_null_as_empty_string")]
    pub raw_content: String,
}

fn deserialize_null_as_empty_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s: Option<String> = Option::deserialize(deserializer)?;
    Ok(s.unwrap_or_default())
}

#[derive(Debug, Serialize, Default, Clone)]
pub struct TavilySearchTool {
    pub tool: BaseTool,
    pub max_results: usize,
    pub api_key: String,
}

impl TavilySearchTool {
    pub fn new(max_results: usize, api_key: Option<String>) -> Self {
        let api_key = if let Some(key) = api_key {
            key
        } else {
            std::env::var("TAVILY_API_KEY").expect("TAVILY_API_KEY is not set")
        };
        TavilySearchTool {
            tool: BaseTool {
                name: "tavily_search",
                description: "Performs a Tavily web search optimized for AI agents, returning relevant search results with content snippets.",
            },
            max_results,
            api_key,
        }
    }

    pub async fn forward(&self, query: &str) -> Result<TavilySearchResponse> {
        let client = reqwest::Client::new();
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let body = json!({
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": false,
            "include_raw_content": false,
            "max_results": self.max_results,
            "include_images": false
        });


        let response = client
            .post("https://api.tavily.com/search")
            .headers(headers)
            .json(&body)
            .send()
            .await?;


        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Tavily API request failed with status {}: {}",
                status,
                error_text
            ));
        }

        let response = response.json::<TavilySearchResponse>().await?;
        Ok(response)
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
        let query = arguments.query;
        let results = self.forward(&query).await?;
        
        let results_string = results
            .results
            .iter()
            .map(|r| {
                format!(
                    "[{}]({}) (Score: {:.2})\n{}\n{}",
                    r.title,
                    r.url,
                    r.score,
                    r.content,
                    if !r.raw_content.is_empty() && r.raw_content != r.content {
                        format!("\nRaw Content: {}", r.raw_content.chars().take(500).collect::<String>())
                    } else {
                        String::new()
                    }
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        if results_string.is_empty() {
            return Err(anyhow::anyhow!("No results found for query: {}", query));
        }

        Ok(results_string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tavily_search_tool() {
        let tool = TavilySearchTool::new(3, None);
        let query = "What is the capital of France?";
        let result = Tool::forward(
            &tool,
            TavilySearchToolParams {
                query: query.to_string(),
            },
        )
        .await
        .unwrap();
        println!("{}", result);
        assert!(result.contains("Paris"));
    }

    #[tokio::test]
    async fn test_tavily_search_tool_with_api_key() {
        // Test with explicit API key
        let api_key = std::env::var("TAVILY_API_KEY").ok();
        if let Some(key) = api_key {
            let tool = TavilySearchTool::new(2, Some(key));
            let query = "AI agents and LLMs";
            let result = Tool::forward(
                &tool,
                TavilySearchToolParams {
                    query: query.to_string(),
                },
            )
            .await;
            
            match result {
                Ok(res) => {
                    println!("Search results: {}", res);
                    assert!(!res.is_empty());
                }
                Err(e) => {
                    println!("Search failed: {}", e);
                    // Don't fail the test if API key is invalid or API is down
                }
            }
        }
    }
}
