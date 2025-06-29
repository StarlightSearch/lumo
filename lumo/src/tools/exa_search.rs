//! This module contains the DuckDuckGo search tool.

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::base::BaseTool;
use super::tool_traits::Tool;
use anyhow::Result;

#[derive(Deserialize, JsonSchema)]
#[schemars(title = "ExaSearchToolParams")]
pub struct ExaSearchToolParams {
    #[schemars(description = "The query to search for")]
    query: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct ExaSearchResponse {
    pub results: Vec<ExaSearchResult>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ExaSearchResult {
    #[serde(default, deserialize_with = "deserialize_null_as_empty_string")]
    pub title: String,
    #[serde(default, deserialize_with = "deserialize_null_as_empty_string")]
    pub url: String,
    #[serde(default, deserialize_with = "deserialize_null_as_empty_string")]
    pub author: String,
    #[serde(default, deserialize_with = "deserialize_null_as_empty_string")]
    pub text: String,
    #[serde(default, deserialize_with = "deserialize_null_as_empty_string")]
    pub summary: String,
}

fn deserialize_null_as_empty_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s: Option<String> = Option::deserialize(deserializer)?;
    Ok(s.unwrap_or_default())
}

#[derive(Debug, Serialize, Default, Clone)]
pub struct ExaSearchTool {
    pub tool: BaseTool,
    pub max_results: usize,
    pub api_key: String,
}

impl ExaSearchTool {
    pub fn new(max_results: usize, api_key: Option<String>) -> Self {
        let api_key = if let Some(key) = api_key {
            key
        } else {
            std::env::var("EXA_API_KEY").expect("EXA_API_KEY is not set")
        };
        ExaSearchTool {
            tool: BaseTool {
                name: "exa_search",
                description: "Performs a exa web search for your query then returns a string of the top search results.",
            },
            max_results,
            api_key,
        }
    }
    pub async fn forward(&self, query: &str) -> Result<ExaSearchResponse> {
        let client = reqwest::Client::new();
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.api_key).expect("Invalid API key"),
        );

        let body = json!({
            "query": query,
            "contents": {
                "text": true
            }
        });

        let response = client
            .post("https://api.exa.ai/search")
            .headers(headers)
            .json(&body)
            .send()
            .await?;

        let response = response.json::<ExaSearchResponse>().await?;
        Ok(response)
    }
}

#[async_trait]
impl Tool for ExaSearchTool {
    type Params = ExaSearchToolParams;
    fn name(&self) -> &'static str {
        self.tool.name
    }
    fn description(&self) -> &'static str {
        self.tool.description
    }
    async fn forward(&self, arguments: ExaSearchToolParams) -> Result<String> {
        let query = arguments.query;
        let results = self.forward(&query).await?;
        let results_string = results
            .results
            .iter()
            .map(|r| {
                format!(
                    "[{}]({}) \n{} \n{}",
                    r.title.clone(),
                    r.url.clone(),
                    r.text.clone(),
                    r.summary.clone()
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
    async fn test_exa_search_tool() {
        let tool = ExaSearchTool::new(2, None);
        let query = "What is the capital of France?";
        let result = Tool::forward(
            &tool,
            ExaSearchToolParams {
                query: query.to_string(),
            },
        )
        .await
        .unwrap();
        println!("{}", result);
        assert!(result.contains("Paris"));
    }
}
