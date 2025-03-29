

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![](https://res.cloudinary.com/dltwftrgc/image/upload/c_fill,h_350,q_auto,r_30,w_1572/v1742300613/Gemini_Generated_Image_bg0xbfbg0xbfbg0x_ha3fs5.png)

A powerful autonomous agent framework written in Rust that leverages LLMs to solve complex tasks using tools and reasoning.

[Installation](#-quick-start) • [Documentation](#-usage) • [Examples](#-examples)

</div>

## 🎯 Overview

Lumo is a Rust implementation of the [smolagents](https://github.com/huggingface/smolagents) library, designed to create intelligent agents that can autonomously solve complex tasks. Built with performance and safety in mind, it provides a robust framework for building AI-powered applications.

### Why Lumo?

- 🚀 **High Performance**: Written in Rust for maximum speed and reliability
- 🛡️ **Type Safety**: Leverages Rust's type system for compile-time guarantees
- 🔄 **Interactive Mode**: Run tasks interactively with real-time feedback
- 🎯 **Task-Oriented**: Focused on solving real-world problems with AI
- 🔌 **Extensible**: Easy to add new tools and integrate with different LLM providers

## ✨ Features

- 🧠 **Function-Calling and CodeAgent**: Implements the ReAct and CodeACT framework for advanced reasoning and action
- 🔍 **Built-in Tools**:
  - Google Search
  - DuckDuckGo Search
  - Website Visit & Scraping
  - Python Interpreter
- 🤝 **Multiple Model Support**: Works with OpenAI, Ollama, and Gemini models
- 🎯 **Task Execution**: Enables autonomous completion of complex tasks
- 🔄 **State Management**: Maintains persistent state across steps
- 📊 **Beautiful Logging**: Offers colored terminal output for easy debugging

---


## ✅ Feature Checklist

### Models

- [x] OpenAI Models (e.g., GPT-4o, GPT-4o-mini)
- [x] Ollama Integration
- [x] Gemini Integration
- [ ] Anthropic Claude Integration
- [ ] Hugging Face API support
- [ ] Open-source model integration via Candle 

You can use models like Groq, TogetherAI using the same API as OpenAI. Just give the base url and the api key.

### Agents

- [x] Tool-Calling Agent
- [x] CodeAgent
- [x] MCP Agent
- [x] Planning Agent
- [x] Multi-Agent Support


### Tools

- [x] Google Search Tool
- [x] DuckDuckGo Tool
- [x] Website Visit & Scraping Tool
- [x] Python Interpreter Tool
- [ ] RAG Tool
- More tools to come...

### Other

- [ ] E2B Sandbox
- [ ] Streaming output
- [ ] Improve logging
- [ ] Tracing

---

## 🚀 Quick Start

### CLI Usage

Warning: Since there is no implementation of a Sandbox environment, be careful with the tools you use. Preferably run the agent in a controlled environment using a Docker container.

#### From Source
1. Build
```bash
cargo build --release
```

2. Run
```bash
target/release/lumo
```

You'll be prompted to enter your task interactively. Type 'exit' to quit the program.

You need to set the API key as an environment variable or pass it as an argument.

You can add the binary to your path to access it from your terminal using `lumo` command. 

#### Using Docker

```bash
# Pull the image
docker pull akshayballal95/lumo-cli:latest

# Run with your API key
docker run -it -e OPENAI_API_KEY=your-key-here lumo-cli
```

The default model is Gemini-2.0-Flash

---

## 🛠️ Usage

```bash
lumo [OPTIONS]

Options:
  -t, --task <TASK>          The task to execute
  -a, --agent-type <TYPE>    Agent type. Options: function-calling, code [default: function-calling]
  -l, --tools <TOOLS>        Comma-separated list of tools. Options: google-search, duckduckgo, visit-website, python-interpreter [default: duckduckgo,visit-website]
  -m, --model <TYPE>         Model type [default: open-ai]
  -k, --api-key <KEY>        LLM Provider API key (only required for OpenAI model)
  --model-id <ID>            Model ID (e.g., "gpt-4" for OpenAI or "qwen2.5" for Ollama) [default: gpt-4o-mini]
  -s, --stream               Enable streaming output
  -b, --base-url <URL>       Base URL for the API [default: https://api.openai.com/v1/chat/completions]
  -h, --help                 Print help
```


## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (optional, if using OpenAI model)
- `GEMINI_API_KEY`: Your Gemini API key (optional, if using Gemini model)
- `SERPAPI_API_KEY`: Google Search API key (optional, if using Google Search Tool)

### Server Configuration

You can configure multiple servers in the configuration file for MCP agent usage. The configuration file location varies by operating system:

- **Linux**: `~/.config/lumo-cli/servers.yaml`
- **macOS**: `~/Library/Application Support/lumo-cli/servers.yaml`
- **Windows**: `%APPDATA%\Roaming\lumo\lumo-cli\servers.yaml`

Example configuration:
```yaml
exa-search:
  command: npx
  args:
    - "exa-mcp-server"
  env: 
    EXA_API_KEY: "your-api-key"

fetch:
  command: uvx
  args:
    - "mcp_server_fetch"

system_prompt: |-
  You are a powerful agentic AI assistant...
```

---

## 🖥️ Server Usage

Lumo can also be run as a server, providing a REST API for agent interactions.

### Starting the Server

#### Using Binary
```bash
# Start the server (default port: 8080)
lumo-server
```

#### Using Docker
```bash
# Build the image
docker build -f server.Dockerfile -t lumo-server .

# Run the container with required API keys
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=your-openai-key \
  -e GOOGLE_API_KEY=your-google-key \
  -e GROQ_API_KEY=your-groq-key \
  -e ANTHROPIC_API_KEY=your-anthropic-key \
  -e EXA_API_KEY=your-exa-key \
  lumo-server
```

You can also use the pre-built image:
```bash
docker pull akshayballal95/lumo-server:latest
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8080/health_check
```

#### Run Task
```bash
curl -X POST http://localhost:8080/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "What is the weather in London?",
    "model": "gpt-4o-mini",
    "base_url": "https://api.openai.com/v1/chat/completions",
    "tools": ["DuckDuckGo", "VisitWebsite"],
    "max_steps": 5,
    "agent_type": "function-calling"
  }'
```

#### Request Body Parameters

- `task` (required): The task to execute
- `model` (required): Model ID (e.g., "gpt-4", "qwen2.5", "gemini-2.0-flash")
- `base_url` (required): Base URL for the API
- `tools` (optional): Array of tool names to use
- `max_steps` (optional): Maximum number of steps to take
- `agent_type` (optional): Type of agent to use ("function-calling" or "mcp")
- `history` (optional): Array of previous messages for context

The server automatically detects the appropriate API key based on the base_url:
- OpenAI URLs use `OPENAI_API_KEY`
- Google URLs use `GOOGLE_API_KEY`
- Groq URLs use `GROQ_API_KEY`
- Anthropic URLs use `ANTHROPIC_API_KEY`

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## ⭐ Show Your Support

Give a ⭐️ if this project helps you or inspires your work!

