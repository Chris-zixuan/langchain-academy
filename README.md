![LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba1020525eea7873f96_LCA-big-green%20(2).svg)

## 介绍

欢迎来到 LangChain Academy —— LangGraph 入门课程！
这是一套不断扩展的模块，专注于 LangChain 生态系统中的基础概念。
模块 0 是基础环境搭建，模块 1-5 专注于使用 LangGraph 构建应用，逐步引入更高级的主题。模块 6 涉及部署你的 Agent。
在每个模块文件夹中，你会看到一组 notebook。每个 notebook 顶部都有一个 LangChain Academy 课程链接，引导你学习相应主题。每个模块还包含一个 `studio` 子目录，其中包含一系列相关图，我们将使用 LangGraph API 和 Studio 进行探索。

## 环境搭建

### Python 版本

请确保你使用的是 Python 3.11、3.12 或 3.13 版本。
```
python3 --version
```

### 克隆仓库
```
git clone https://github.com/langchain-ai/langchain-academy.git
$ cd langchain-academy
```
或者，如果你更喜欢，可以在[这里](https://github.com/langchain-ai/langchain-academy/archive/refs/heads/main.zip)下载 zip 文件。

### 创建虚拟环境并安装依赖
#### Mac/Linux/WSL
```
$ python3 -m venv lc-academy-env
$ source lc-academy-env/bin/activate
$ pip install -r requirements.txt
```
#### Windows Powershell
```
PS> python3 -m venv lc-academy-env
PS> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
PS> .\lc-academy-env\Scripts\Activate.ps1
PS> pip install -r requirements.txt
```

### 运行 Notebook
如果你还没有安装 Jupyter，请按照[这里](https://jupyter.org/install)的安装说明操作。
```
$ jupyter notebook
```

### 设置环境变量
简要说明如何设置环境变量。
#### Mac/Linux/WSL
```
$ export API_ENV_VAR="your-api-key-here"
```
#### Windows Powershell
```
PS> $env:API_ENV_VAR = "your-api-key-here"
```

### 设置 DeepSeek API 密钥
* 如果你还没有 DeepSeek API 密钥，可以在[这里](https://platform.deepseek.com/)注册。
* 在环境变量中设置 `DEEPSEEK_API_KEY`

### 注册并设置 LangSmith API
* 在[这里](https://docs.langchain.com/langsmith/create-account-api-key#create-an-account-and-api-key)注册 LangSmith，在[这里](https://www.langchain.com/langsmith)了解更多关于 LangSmith 以及如何在工作流中使用它的信息。
* 在环境变量中设置 `LANGSMITH_API_KEY`、`LANGSMITH_TRACING_V2="true"`、`LANGSMITH_PROJECT="langchain-academy"`
* 如果你使用的是 EU 实例，还需要设置 `LANGSMITH_ENDPOINT`="https://eu.api.smith.langchain.com"。

### 设置 Tavily API（用于网络搜索）

* Tavily Search API 是一个针对 LLM 和 RAG 优化的搜索引擎，旨在提供高效、快速且持久的搜索结果。
* 你可以在[这里](https://tavily.com/)注册 API 密钥。
注册非常简单，并提供非常慷慨的免费额度。部分课程（模块 4）将使用 Tavily。

* 在环境变量中设置 `TAVILY_API_KEY`。

### 设置 Studio

* Studio 是一个用于查看和测试 Agent 的定制 IDE。
* Studio 可以在 Mac、Windows 和 Linux 上本地运行，并在浏览器中打开。
* 关于本地 Studio 开发服务器，请参阅[这里](https://docs.langchain.com/langsmith/studio#local-development-server)的文档。
* LangGraph Studio 的图位于模块 1-5 的 `module-x/studio/` 文件夹中。
* 要启动本地开发服务器，请确保你的虚拟环境已激活，并在每个模块的 `/studio` 目录下的终端中运行以下命令：

```
langgraph dev
```

你应该会看到以下输出：
```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

打开浏览器并导航到 Studio UI：`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`。

* 要使用 Studio，你需要创建一个包含相关 API 密钥的 .env 文件
* 以模块 1 到 5 为例，在命令行中运行以下命令来创建这些文件：
```
for i in {1..5}; do
  cp module-$i/studio/.env.example module-$i/studio/.env
  echo "DEEPSEEK_API_KEY=\"$DEEPSEEK_API_KEY\"" > module-$i/studio/.env
done
echo "TAVILY_API_KEY=\"$TAVILY_API_KEY\"" >> module-4/studio/.env
```
