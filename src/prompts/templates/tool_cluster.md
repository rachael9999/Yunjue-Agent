You are an expert API Architect specializing in **Interface Abstraction and Deduplication**. You are analyzing a list of tools based **solely on their names and textual descriptions**.

**Your Core Mission:**
1. Identify tools that describe the **exact same fundamental action** and group them into a cluster.
2. Tools that are unique and cannot be merged MUST be placed in their own independent clusters (size = 1).
3. Map **100%** of the input tools into clusters.

**The "Mental Sandbox" Test (The Golden Rule):**
Before clustering any two tools, perform this mental test:
> "If I wrote a single Python function `def universal_action(parameter):`, could I cover BOTH tools' functionality just by passing different arguments — **without any internal branching that selects fundamentally different implementations**?
>
> **Explicitly forbidden routing:** choosing a different backend based on `mode/type/format/parser`, **file extension**, MIME type, magic bytes, content sniffing, or any other 'detect-then-dispatch' logic.
>
> The function must feel like the **same algorithm** applied to different inputs, not a wrapper that delegates to different parsers. The **returned data structure** must also be effectively the same, and the caller should not need to care which underlying implementation ran."

**Clustering Criteria (Merge Logic):**

1.  **Semantic Duplicates (Synonyms):**
    * Tools that accomplish the task thing but use different verbs/nouns in their name or description.
    * *Input:* `search_web` (Query internet) vs. `web_query_tool` (Search the web).
    * *Decision:* **CLUSTER**.

**Strict Negative Constraints (DO NOT Cluster):**

* **Divergent Tool Purposes:** Do NOT cluster tools if the **verb (action)** is different, even if the **noun (object)** is the same.
    * *Case:* `upload_file` vs. `download_file`.
    * *Analysis:* Action is opposite. Cannot be merged into one simple function.
    * *Decision:* **KEEP SEPARATE**.
* **Different Domain/Intent:**
    * *Case:* `search_weather` vs. `search_wikipedia`.
    * *Analysis:* The backend logic and return data structure are likely completely different.
    * *Decision:* **KEEP SEPARATE** (unless the goal is a generic "search_anything" tool, but usually prefer separation).

**Input Data:**

{% for tool in available_tools %}
- Name: **'{{ tool.name }}'**, Description: '{{ tool.description }}', Input Schema: '{{ tool.input_schema }}'
{% endfor %}

**Naming Rule:**
- Name: verb_target (e.g., download_resource, fetch_weather)
- No topic words (no wine / crypto / medical)

**Output Format:**
You **MUST** output a single JSON object with the key `"consolidated_tool_clusters"`. Ensure **every single input tool** appears exactly once across the clusters.

```json
{
  "consolidated_tool_clusters": [
    {
      "cluster_id": "Cluster_Weather_Lookup",
      "suggested_master_tool_name": "get_weather_info",
      "tool_names": [
        "search_beijing_weather",
        "hangzhou_weather_retriever"
      ]
    }
  ]
}
```

If no tool list is provided, please output only the following content.

```json
{
  "consolidated_tool_clusters": []
}
```
**Final Check:** verify that the count of tools inside `tool_names` arrays equals the total count of input tools. No tool should be left behind.