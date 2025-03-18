

![](https://res.cloudinary.com/dltwftrgc/image/upload/c_fill,h_350,q_auto,r_30,w_1572/v1742300613/Gemini_Generated_Image_bg0xbfbg0xbfbg0x_ha3fs5.png)

---

Agentic library to make high-performance CLI and server agents, with seamless integration with MCP.


## ✨ Features

-  **Function-Calling Agent Architecture**: Implements the ReAct framework for advanced reasoning and action.
-  **Built-in Tools**:
-  **OpenAI Integration**: Works seamlessly with GPT models.
-  **Task Execution**: Enables autonomous completion of complex tasks.
-  **State Management**: Maintains persistent state across steps.
-  **Beautiful Logging**: Offers colored terminal output for easy debugging.

---

![demo](https://res.cloudinary.com/dltwftrgc/image/upload/v1737485304/smolagents-small_fmaikq.gif)

## ✅ Feature Checklist

### Models

- [x] OpenAI Models (e.g., GPT-4o, GPT-4o-mini)
- [x] Ollama Integration
- [ ] Hugging Face API support
- [ ] Open-source model integration via Candle 

You can use models like Groq, TogetherAI using the same API as OpenAI. Just give the base url and the api key.

### Agents

- [x] Tool-Calling Agent
- [x] CodeAgent
- [x] Planning Agent
- [x] Multi-Agent Support

The code agent is still in development, so there might be python code that is not yet supported and may cause errors. Try using the tool-calling agent for now.

### Tools

- [x] Google Search Tool
- [x] DuckDuckGo Tool
- [x] Website Visit & Scraping Tool
- [ ] RAG Tool
- More tools to come...

### Other

- [ ] E2B Sandbox
- [ ] Streaming output
- [ ] Improve logging

---

## 🚀 Quick Start

### CLI Usage

Warning: Since there is no implementation of a Sandbox environment, be careful with the tools you use. Preferrably run the agent in a controlled environment using a Docker container.

#### Using Cargo

```bash
cargo install lumo --all-features
```

```bash
lumo -t "Your task here"
```
You need to set the API key as an environment variable. Otherwise you can pass it as an argument.

#### Using Docker

```bash
# Pull the image
docker pull your-username/lumo:latest

# Run with your OpenAI API key
docker run -e OPENAI_API_KEY=your-key-here lumo -t "What is the latest news about Rust programming?"
```

---


## 🛠️ Usage

```bash
lumo [OPTIONS] -t TASK

Options:
  -a, --agent-type <AGENT_TYPE>  The type of agent to use [default: function-calling] [possible values:
                                 function-calling, code, mcp]
  -l, --tools <TOOLS>...         List of tools to use [default: duck-duck-go visit-website] [possible values:
                                 duck-duck-go, visit-website, google-search-tool, python-interpreter]
  -m, --model-type <MODEL_TYPE>  The type of model to use [default: open-ai] [possible values: open-ai, ollama]
  -k, --api-key <API_KEY>        OpenAI API key (only required for OpenAI model)
      --model-id <MODEL_ID>      Model ID (e.g., "gpt-4" for OpenAI or "qwen2.5" for Ollama) [default: gpt-4o-mini]
  -p, --planning-interval        Number of steps to take to update the plan
  -b, --base-url <BASE_URL>      Base URL for the API
      --max-steps <MAX_STEPS>    Maximum number of steps to take [default: 10]
  -h, --help                     Print help
  -V, --version                  Print version
```

---

## 🌟 Examples

```bash
# Simple search task
lumo -t "What are the main features of Rust 1.75?"

# Research with multiple tools
lumo -t "Compare Rust and Go performance" -l duckduckgo,google-search,visit-website

# Stream output for real-time updates
lumo -t "Analyze the latest crypto trends" -s
```
---

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (optional, if you want to use OpenAI model).
- `SERPAPI_API_KEY`: Google Search API key (optional, if you want to use Google Search Tool).


## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add some amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.


---

## ⭐ Show Your Support

Give a ⭐️ if this project helps you or inspires your work!

