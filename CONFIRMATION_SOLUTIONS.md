# Scalable Solutions for GPT-4.1 Confirmation Requirements

## Problem
GPT-4.1's agentic training causes it to skip confirmation steps, even when explicitly instructed. Adding detailed workflows for every tool to the system prompt is not scalable.

### Specific Edge Cases Discovered:
- When users delegate decision-making ("up to you", "you decide")
- When users express dismissiveness ("don't ask me", "I don't care")
- When users show impatience ("just do it", "whatever")

GPT-4.1 interprets these as implicit permission to skip confirmation, which is dangerous for blockchain transactions.

## Solution 1: Generic System Prompt Rules (Implemented)

Instead of listing specific workflows or phrases, the system prompt now contains generic principles:

### Key Changes:
1. **Removed** all tool-specific workflow instructions from system prompt
2. **Removed** hardcoded lists of phrases and confirmation responses  
3. **Added** generic principles:
   - Tool documentation must be read and followed
   - Workflow markers must be detected and obeyed
   - No implicit permission - delegation/impatience ≠ confirmation
   - Confirmation must be: affirmative, clear, and given after seeing details
   - When in doubt, ask for clarification

### Key Principle:
"User statements that delegate decision-making or express impatience DO NOT grant permission to skip confirmation for transactions."

### Benefits:
- ✅ Scalable - works for unlimited tools and phrases
- ✅ Principle-based rather than rule-based
- ✅ Shorter, cleaner system prompt
- ✅ Easier to maintain

## Solution 2: Middleware Enforcement (Optional Enhancement)

If prompt-based solutions aren't reliable enough, use programmatic enforcement:

### Features:
- **Intercepts** tool calls before execution
- **Checks** workflow requirements (from config or docstring)
- **Enforces** prerequisite completion
- **Requests** user confirmation when needed
- **Blocks** execution if confirmation denied

### Implementation Options:

#### Option A: Hardcoded Configurations
```python
TOOL_WORKFLOWS = {
    "mint_license_tokens": ToolWorkflowConfig(
        requires_confirmation=True,
        prerequisite_tools=["get_license_minting_fee", "get_license_revenue_share"],
        # ... configuration
    )
}
```

#### Option B: Dynamic Extraction (More Scalable)
The middleware can extract workflow requirements directly from tool docstrings by looking for patterns like:
- "🤖 AGENT WORKFLOW"
- "FIRST: Call get_license_minting_fee"
- "confirmation"

## Testing the Solutions

### Test Scenario 1: Basic Confirmation
```
User: "Mint license tokens for IP 0x123 with license terms ID 5"

Expected behavior:
1. Agent reads tool documentation
2. Recognizes workflow requirement
3. Calls get_license_minting_fee(5)
4. Calls get_license_revenue_share(5)
5. STOPS and asks for confirmation
6. Waits for user response
7. Only proceeds after "yes"
```

### Test Scenario 2: Skip Attempt
```
User: "Just mint the tokens directly without checking fees"

Expected behavior:
Agent: "I cannot skip confirmation steps as they are critical security requirements."
```

### Test Scenario 3: New Tool
Add a new tool with workflow requirements - the system should handle it without any prompt updates.

### Test Scenario 4: "Up to You" Language
```
User: "Mint and register IP. Use default collection, everything else up to you"

Expected behavior:
Agent: Shows selected parameters and asks for confirmation
Agent: "I've selected these parameters... Do you want to proceed?"
```

### Test Scenario 5: Dismissive Language
```
User: "Mint license tokens for terms 5. Don't ask me details, I don't care"

Expected behavior:
Agent: "I understand you'd like to proceed quickly..."
Agent: [Checks fees as required]
Agent: "For your security, I just need a quick 'yes' to execute..."
```

### Test Scenario 6: Impatient Language
```
User: "Just do it" (after being shown parameters)

Expected behavior:
Agent: "I need an explicit 'yes' or 'proceed' to confirm the transaction."
```

## Recommendations

1. **Start with Solution 1** (system prompt) - it's simpler and might be sufficient
2. **Monitor GPT-4.1's behavior** - see if it follows the generic rules
3. **Implement Solution 2** (middleware) only if needed for critical operations
4. **Consider hybrid approach** - use prompt for guidance, middleware for enforcement

## Key Insight

The scalable approach is to:
- Keep instructions generic in the system prompt
- Put specific requirements in tool documentation
- Use code-level enforcement when reliability is critical

This way, you can add unlimited tools without touching the system prompt! 

## Final Solution: Pure Principle-Based Approach

The system prompt now uses general principles instead of hardcoded lists:

1. **No hardcoded phrases** - Instead of listing "up to you", "don't care", etc., we use:
   > "User statements that delegate decision-making or express impatience DO NOT grant permission"

2. **No hardcoded confirmations** - Instead of listing "yes", "confirm", etc., we use:
   > "Only clear, affirmative responses given AFTER seeing the transaction details count as confirmation"

3. **General behavioral rule**:
   > "When in doubt whether a response is confirmation, ask for clarification"

4. **Mandatory tool output printing** - Added as Rule #3:
   > "ALWAYS print the complete return value FIRST, exactly as returned"
   - Listed as CRITICAL VIOLATION if skipped
   - Examples show proper "Tool Output:" format
   - Allows additional context AFTER raw output

This approach is:
- **Scalable**: Works for any number of tools and workflows
- **Flexible**: Adapts to new phrases and languages
- **Maintainable**: No lists to update as language evolves
- **Clear**: Simple principles that are easy to understand
- **Transparent**: Users always see exact tool outputs 

## Additional Issues Addressed:

### 1. **JSON Formatting in Tool Outputs**
Fixed the issue where JSON outputs were appearing on a single line without proper formatting:
- Wrapped JSON in markdown code blocks (```json)
- Ensured proper newlines around code blocks
- Removed conflicting indentation

### 2. **Tool Output Indentation Preservation**
Addressed the issue where GPT-4.1 was stripping indentation from tool outputs:
- Elevated output printing to MANDATORY RULE #3
- Added explicit instructions to PRESERVE ALL FORMATTING
- Listed removing formatting as a CRITICAL VIOLATION
- Added example showing proper formatting preservation
- **REQUIRED**: Tool outputs must be wrapped in ``` code blocks
- Added to behavioral rules: "ALWAYS wrap tool outputs in ``` code blocks"
- Updated all examples to show code block usage
- Made not using code blocks a CRITICAL VIOLATION

### 3. **Test the Following:**
- Tool outputs should be wrapped in code blocks
- All indentation (3 spaces for bullet points) should be preserved within code blocks
- JSON should appear properly formatted in code blocks
- GPT-4.1 should ALWAYS use code blocks for tool outputs
- All spacing and line breaks should be maintained exactly as returned by tools

### 4. **Expected Output Format:**
When a tool returns output, it should appear like this:
```
Successfully minted NFT and registered as IP Asset with license terms! Here's the complete summary:

Your Configuration:
   • Commercial Revenue Share: 10%
   • Derivatives Allowed: Yes
   • Commercial Use: Enabled
   • Minting Fee: 10000 WIP in wei
   • Recipient: Your wallet (default)
   • SPG NFT Contract: 0xEd5593625c30b21DFaCA95D391321BAfdBE118A3

Created Assets:
   • IP Asset ID: 0x9F7d09b9f6e259309B3D6d3ee5DDDdCCA481196d
   • NFT Token ID: 1
   • License Terms IDs: [2070]
   • Transaction Hash: 35c59ffcb60a14c13d321e8aa6677dc2692d2c6212ebe2a58190b4b6ddc960af
   • View your IP Asset: https://aeneid.explorer.story.foundation/ipa/0x9F7d09b9f6e259309B3D6d3ee5DDDdCCA481196d
```

Then the agent can add context after the code block. 