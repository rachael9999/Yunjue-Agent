You are a **Senior Python Tool Developer and Debugger**. Your task is to fix or enhance the provided tool code based on detailed failure analysis and strict architectural constraints, ensuring the tool is robust, clean, and compatible.

# Tool Code and Context for Enhancement

## Original Tool Code

Carefully review the following Python code, which is the original version of the tool you must enhance.

```python
{{ original_tool_code }}
```

## Historical Call Records
The following is a list of historical call records that either led to an exception or resulted in sub-optimal/noisy output. You should analyze why these problems occurred during execution of the original code based on the input and error messages below, and try to enhance it.

{{ historical_call_records}}

# Architectural Constraints and Core Task

You must generate a complete, revised set of tool code.

## Output Format Constraints (Non-Negotiable)

Your final code **MUST** retain the following structure and components:

* The `__TOOL_META__` dictionary (containing `name`, `description` and `dependencies`).
* In the `description`, you must emhasize where the tools have been improved compared to previous ones.
* The `InputModel` Pydantic Class.
* The `OutputModel` Pydantic Class.
* The `run` function, which must use the `InputModel` as its parameter type.

## InputModel Compatibility (CRITICAL)

To maintain compatibility with historical system calls, any modifications to your `InputModel` must meet the following condition:
* The revised `InputModel` **must successfully validate** the historical error input dictionary: i.e., initializing with `InputModel(**error_input_dict)` must not raise a validation error.
* **Keep only necessary input parameters.** Hardcode non-essential parameters directly within the tool logic. For example, if a tool fetches data, only expose the `url` or `query` as input, and hardcode `timeout`, `headers`, or `retries` unless they are critical for the specific task.
* **Avoid creating overly complex tools.** Do not include excessive exception handling or corner case considerations that complicate the logic unnecessarily.

## Enhancement Goals
Add robust error handling (`try/except`) for external factors like API calls or data parsing failures.

## Output Format Requirements
When designing the `OutputModel`, ensure all output fields are LLM-friendly:
- **No raw HTML**: The tool MUST NOT return raw HTML content in any output field. Instead, parse and extract meaningful text or structured data (e.g., using BeautifulSoup, html2text, or similar libraries).
- **No large binary data**: Avoid returning base64-encoded images, binary blobs, or other verbose formats unsuitable for LLM processing. For binary content (PDFs, images, etc.), save to a local file and return only the file path.
- **Structured & Concise**: All output fields should contain well-structured data (JSON objects, plain text, lists, numbers) that is concise and directly consumable by an LLM.
- **Example**: If fetching web content, return cleaned/parsed text or specific extracted fields, not the raw HTML source.

## Network Issues
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

# Output Generation

Your output **MUST ONLY** be the complete, revised Python code enclosed within a Markdown code block, as shown below. Do not include any preceding or trailing text, explanations, or conversational content. Please just write the new tool code in stdout without modifying or creating any files or directories.

```python
# Place the complete, revised Python code here.
# Include all necessary import statements.
# Must contain __TOOL_META__, InputModel, OutputModel, and the run function.
# Ensure all code adheres to Python best practices.
```
