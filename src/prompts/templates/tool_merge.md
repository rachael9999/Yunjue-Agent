You are an expert Python software engineer specializing in code consolidation and refactoring.

**Task:** Merge the following set of Python code snippets into a single, cohesive, and well-organized Python file. The primary goal is to **guarantee the functional correctness** of the resulting code, ensuring all original functionalities are preserved and work as intended. Please just write the new tool code **without** modifying any files or directories in the original directory.

**Keep only necessary input parameters.** Hardcode non-essential parameters directly within the tool logic. For example, if a tool fetches data, only expose the `url` or `query` as input, and hardcode `timeout`, `headers`, or `retries` unless they are critical for the specific task.

**Avoid creating overly complex tools.** Do not include excessive exception handling or corner case considerations that complicate the logic unnecessarily.

**Input Code Snippets:**
{% for tool in tools_code %}
=============== The {{tool.idx}}th Tool {{tool.name}} Begin ==================
{{tool.code}}
=============== The {{tool.idx}}th Tool {{tool.name}} End ===================
{% endfor %}

**Network Issues**
**Allowed API Keys:** You can construct tools with only the `TAVILY_API_KEY` API key. **Do NOT** construct tools using any other API keys.
    * Your code MUST load these keys using `os.environ.get("KEY_NAME")`. **Never hardcode keys.**

{% if proxy_url %}**Proxy Configuration:** When accessing external networks, configure HTTP/HTTPS requests to use the proxy `{{ proxy_url }}` directly in the code. **Do NOT** include `proxy` parameters in `InputModel`. Example:

```python
with httpx.Client(proxy="{{ proxy_url }}", timeout=timeout) as client:
    resp = client.request(method, url, headers=headers, json=json)
    resp.raise_for_status()
    return resp.json()
```

Bad Example (Do NOT do this):

```python
class InputModel(BaseModel):
    proxy: Optional[str] = Field(
        None,
        description="Optional proxy URL override (e.g., {{ proxy_url }}). If not set, environment proxies may apply.",
    )
    ......
```
{% endif %}

**URL fetching**: If this `description` is about **URL fetching**, you **MUST use Crawl4ai**. Crawl4ai will fetch the page and convert it into Markdown; you only need to use `result.markdown` to get Markdown. This kind of URL-fetching tool should **NOT** fetch binary / media-only content (e.g. PDFs, images, videos) because the returned result is meant to be read by an LLM, and the LLM cannot directly understand base64/binary payloads.
    * State explicitly in the description that the tool can only fetch web pages (Markdown) and must not download files.
    * **MANDATORY (Crawl4AI usage):** When using Crawl4AI, you **MUST** (a) instantiate the crawler `AsyncWebCrawler` with `magic=True` and `config=BrowserConfig(user_agent=xxx, headers=xxx)` and (b) wrap the crawl in `contextlib.redirect_stdout(io.StringIO())` to capture/suppress noisy stdout logs. Do not omit either of these.
    * Here is a correct example about usage of Crawl4ai (follow this pattern exactly):
```python
import asyncio
import io
import contextlib
from crawl4ai import AsyncWebCrawler, BrowserConfig
async def main():
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        async with AsyncWebCrawler(magic=True, config=BrowserConfig(user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/116.0.0.0 Safari/537.36")) as crawler:
            result = await crawler.arun(
                url="https://google.com",
            )
            content = result.markdown
```

**Downloading file**: If this `description` is about **downloading content from a URL to local files**, you should use anti-bot / anti-scraping techniques (e.g., realistic headers, randomized delays, retries/backoff, cookie/session handling where appropriate), and the tool must **never** use **Crawl4ai**. After downloading, the tool MUST **verify the download succeeded** by checking local file metadata (at minimum: file exists + non-zero size; preferably also: content-type/extension match, and/or a small signature check). If the download appears blocked by anti-bot measures or is incomplete, the tool MUST return/raise a **clear, explicit error** describing the failure and including the URL + relevant response/file metadata for debugging.

**Output Format Constraints (Non-Negotiable)**

Your final code **MUST** retain the following structure and components:

* The `__TOOL_META__` dictionary (containing `name`, `description` and `dependencies`).
* In the `description`, only describe the functionality of the merged tool. Do not include statements like "This tool is a merge of tool A and tool B".
* In the `name`, you should use {{ suggest_name }}.
* The `InputModel` Pydantic Class.
* The `OutputModel` Pydantic Class.
* The `run` function, which must use the `InputModel` as its parameter type.

Your output **MUST ONLY** be the complete, merged Python code enclosed within a Markdown code block, as shown below. Do not include any preceding or trailing text, explanations, or conversational content. **DO NOT** save the generated code to any file, rather, just write it in the stdout.

```python
# Place the complete, revised Python code here.
# Include all necessary import statements.
# Must contain __TOOL_META__, InputModel, OutputModel, and the run function.
# Ensure all code adheres to Python best practices.
```
