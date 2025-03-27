use std::{collections::HashMap, future::Future, path::Path, pin::Pin};
use std::sync::Arc;

use crate::tools::BaseTool;
use anyhow::Result;
use arrow_array::RecordBatch;
use async_trait::async_trait;
use futures::TryStreamExt;
use lancedb::{
    connect,
    index::scalar::FullTextSearchQuery,
    query::{ExecutableQuery, QueryBase},
    DistanceType, Table as LanceDbTable,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::Tool;

#[derive(Serialize, Debug)]
pub struct SearchResponse {
    file_name: String,
    file_path: String,
    text: String,
    score: String,
    id: String,
    page_number: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
#[schemars(title = "LanceRAGToolParams")]
pub struct LanceRAGToolParams {
    #[schemars(description = "The query to search for.")]
    query: String,
}

#[derive(Clone)]
pub struct LanceRAGTool {
    pub tool: BaseTool,
    pub table: LanceDbTable,
    pub embedding_fn: Arc<dyn Fn(String) -> Pin<Box<dyn Future<Output = Vec<f32>> + Send>> + Send + Sync>,
    pub limit: usize,
}

impl LanceRAGTool {
    pub async fn new(
        url: &str,
        table_name: &str,
        embedding_fn: Box<dyn Fn(String) -> Pin<Box<dyn Future<Output = Vec<f32>> + Send>> + Send + Sync>,
        limit: usize,
    ) -> Result<Self> {
        let db = connect(url).execute().await?;
        let table = db.open_table(table_name).execute().await?;
        Ok(LanceRAGTool {
            tool: BaseTool{
                name: "file_search",
                description: "Search for documents in a LanceDB table. Use this tool when you need to search documents and get information.",
            },
            table,
            embedding_fn: Arc::from(embedding_fn),
            limit,
        })
    }

    pub async fn forward(&self, query: &str) -> Result<Vec<SearchResponse>> {
        let limit: usize = self.limit;

        let query_point: Vec<f32> = (self.embedding_fn)(query.to_string()).await;

        let filter = format!("workspace_name == '{}'", "Zotero");

        // Get vector search results with filter
        let vector_search_result: Vec<RecordBatch> = self
            .table
            .vector_search(query_point)?
            .distance_type(DistanceType::Cosine)
            .limit(limit)
            .only_if(&filter)
            .execute()
            .await
            .map_err(|e| anyhow::anyhow!("Error collecting vector search results: {:?}", e))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| anyhow::anyhow!("Error collecting vector search results: {:?}", e))?;

        let text_search_result = match self
            .table
            .query()
            .full_text_search(FullTextSearchQuery::new(query.to_string()))
            .limit(limit)
            .only_if(&filter)
            .execute()
            .await
        {
            Ok(result) => match result.try_collect::<Vec<_>>().await {
                Ok(batches) => batches,
                Err(e) => {
                    println!("Error collecting text search results: {:?}", e);
                    Vec::new()
                }
            },
            Err(e) => {
                println!("Error executing text search: {:?}", e);
                Vec::new()
            }
        };
        let mut combined_results: Vec<SearchResponse> = Vec::new();
        let mut rrf_scores: HashMap<String, f32> = HashMap::new();
        const K: f32 = 60.0;

        for (batch_index, record_batch) in vector_search_result.iter().enumerate() {
            process_batch_for_rrf_scores(batch_index, record_batch, &mut rrf_scores, K);
        }

        for (batch_index, record_batch) in text_search_result.iter().enumerate() {
            process_batch_for_rrf_scores(batch_index, record_batch, &mut rrf_scores, K);
        }

        // Convert results to SearchResponse objects
        for record_batch in vector_search_result.iter().chain(text_search_result.iter()) {
            process_batch_for_results(record_batch, &rrf_scores, &mut combined_results);
        }

        // Sort by RRF score descending
        combined_results.sort_by(|a, b| {
            b.score
                .parse::<f32>()
                .unwrap_or(0.0)
                .partial_cmp(&a.score.parse::<f32>().unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(combined_results.into_iter().take(limit).collect())
    }
}

#[async_trait]
impl Tool for LanceRAGTool {
    type Params = LanceRAGToolParams;
    
    fn name(&self) -> &'static str {
        self.tool.name
    }

    fn description(&self) -> &'static str {
        self.tool.description
    }

    async fn forward(&self, arguments: LanceRAGToolParams) -> Result<String> {
        let query = arguments.query;
        let results = self.forward(&query).await?;
        Ok(results
            .into_iter()
            .enumerate()
            .map(|(i, r)| {
                format!(
                    "\nContext {} \nText: {} --------",
                    i + 1,
                    // r.file_name,
                    // r.page_number.unwrap_or("N/A".to_string()),
                    r.text
                )
            })
            .collect::<Vec<_>>()
            .join("\n"))
    }
}

fn process_batch_for_rrf_scores(
    batch_index: usize,
    record_batch: &RecordBatch,
    rrf_scores: &mut HashMap<String, f32>,
    k: f32,
) {
    for row_index in 0..record_batch.num_rows() {
        let id = record_batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap()
            .value(row_index)
            .to_string();

        let rank = (batch_index * record_batch.num_rows() + row_index + 1) as f32;
        *rrf_scores.entry(id).or_insert(0.0) += 1.0 / (rank + k);
    }
}
fn process_batch_for_results(
    record_batch: &RecordBatch,
    rrf_scores: &HashMap<String, f32>,
    combined_results: &mut Vec<SearchResponse>,
) {
    for row_index in 0..record_batch.num_rows() {
        let id = record_batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap()
            .value(row_index)
            .to_string();

        // Only add if we haven't processed this ID yet

        let page_number = record_batch.column_by_name("page_number").and_then(|col| {
            col.as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .map(|arr| arr.value(row_index).to_string())
        });
        if !combined_results.iter().any(|r| r.id == id) {
            let response = SearchResponse {
                file_name: Path::new(
                    &record_batch
                        .column_by_name("file_name")
                        .unwrap()
                        .as_any()
                        .downcast_ref::<arrow_array::StringArray>()
                        .unwrap()
                        .value(row_index)
                        .to_string(),
                )
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string(),
                file_path: record_batch
                    .column_by_name("file_name")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .unwrap()
                    .value(row_index)
                    .to_string(),
                text: record_batch
                    .column_by_name("text")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .unwrap()
                    .value(row_index)
                    .to_string(),
                page_number,
                score: rrf_scores.get(&id).unwrap_or(&0.0).to_string(),

                id,
            };
            combined_results.push(response);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use embed_anything::embeddings::embed::EmbedderBuilder;

    use super::*;

    #[tokio::test]
    async fn test_lance_rag_tool() {
        let dense_model = Arc::new(
            EmbedderBuilder::new()
                .model_architecture("jina")
                .model_id(Some("jinaai/jina-embeddings-v2-small-en"))
                .path_in_repo(Some("model.onnx"))
                .from_pretrained_onnx()
                .unwrap(),
        );

        let tool = LanceRAGTool::new(
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
            5
        ).await.unwrap();
        let result = Tool::forward(&tool, LanceRAGToolParams {
            query: "What is transformers?".to_string(),
        }).await.unwrap();
        // println!("{}", result);
    }
}
