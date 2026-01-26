Analyze the following information to determine whether the `Worker Response` contains useful information.

# Worker Response
{{ pending_response }}

# Analysis Focus:
1. **RETRY:** Set status to 'RETRY' **ONLY IF** ANY of the following conditions are met:
   - The `Worker Response` contains an explicit "not found" outcome (e.g., **"Information not found"** or an unambiguous equivalent such as "No results found", "Unable to find", "Couldn't find", etc.); **OR**
   - The `Worker Response` explicitly states that **a tool error/failure prevented completing the task** and therefore they **could not proceed / could not complete / could not obtain the required info / could not finish the requested work**; **OR**
   - The `Worker Response` does **NOT** contain a "Final Conclusion" or similar conclusive statement (e.g., "In conclusion", "To summarize", "Final answer", "Summary", etc.), indicating the task is incomplete.
2. **FINISH:** Otherwise, set status to 'FINISH' **as long as** the `Worker Response` contains **any useful information** **AND** includes a conclusive statement.

# Output Format

Output the result in the following JSON format:
```json
{
    "status": "FINISH" or "RETRY",
    "reason": "A short explanation of your decision."
}
```