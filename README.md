

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

- 🧠 **Function-Calling Agent Architecture**: Implements the ReAct framework for advanced reasoning and action
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

#### Using Cargo

```bash
cargo install lumo --all-features
```

```bash
lumo
```

You'll be prompted to enter your task interactively. Type 'exit' to quit the program.

You need to set the API key as an environment variable or pass it as an argument.

#### Using Docker

```bash
# Pull the image
docker pull your-username/lumo:latest

# Run with your API key
docker run -e OPENAI_API_KEY=your-key-here lumo
```

---

## 🛠️ Usage

```bash
lumo [OPTIONS]

Options:
  -a, --agent-type <TYPE>    Agent type. Options: function-calling, code, mcp [default: function-calling]
  -l, --tools <TOOLS>        Comma-separated list of tools. Options: google-search, duckduckgo, visit-website, python-interpreter [default: duckduckgo,visit-website]
  -m, --model-type <TYPE>    Model type. Options: openai, ollama, gemini [default: gemini]
  -k, --api-key <KEY>        LLM Provider API key
  --model-id <ID>            Model ID (e.g., "gpt-4" for OpenAI, "qwen2.5" for Ollama, or "gemini-2.0-flash" for Gemini) [default: gemini-2.0-flash]
  -b, --base-url <URL>       Base URL for the API
  --max-steps <N>            Maximum number of steps to take [default: 10]
  -p, --planning-interval <N> Planning interval
  -v, --logging-level <LEVEL> Logging level
  -h, --help                 Print help
```

---

## 🌟 Examples

```bash
# Basic usage with default settings
lumo

# Using specific model and tools
lumo -m ollama --model-id qwen2.5 -l duckduckgo,google-search

# Using OpenAI with custom base URL
lumo -m openai --model-id gpt-4 -b "https://your-custom-url.com/v1"
```

---

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
  command: python3
  args:
    - "-m"
    - "mcp_server_fetch"

system_prompt: |-
  You are a powerful agentic AI assistant...
```

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

