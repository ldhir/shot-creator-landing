# AI Coach Cost Estimate

## Your $2 OpenAI Credits

**Good news: $2 is enough for testing and light usage!** Here's the breakdown:

### Cost Per Message (GPT-3.5-turbo - Recommended)

- **Input tokens**: ~1,500-2,500 tokens (system prompt + user context + history)
- **Output tokens**: ~200-400 tokens (coach response)
- **Cost per message**: ~$0.002 - $0.004

### With $2, You Can Get:

- **~500-1,000 messages** with GPT-3.5-turbo
- That's enough for:
  - Testing the system thoroughly
  - Daily use for 1-2 months (if used moderately)
  - Multiple users trying it out

### Cost Comparison

| Model | Cost per Message | Messages with $2 |
|-------|-----------------|-------------------|
| GPT-3.5-turbo | ~$0.003 | ~650 messages |
| GPT-4 | ~$0.10 | ~20 messages |

**Recommendation**: The code is set to use GPT-3.5-turbo by default, which is perfect for your budget!

## Tips to Maximize Your Credits

1. **Use GPT-3.5-turbo** (already set as default)
2. **Limit conversation history**: Only last 10 messages are sent (already implemented)
3. **Reduce max_tokens**: Set to 500 (already done) - still plenty for good responses
4. **Monitor usage**: Check your OpenAI dashboard regularly
5. **Clear old conversations**: Use the "Clear Chat" button to reset history

## When You Might Need More Credits

- Heavy daily usage (100+ messages/day)
- Multiple users
- Want to use GPT-4 for better quality (10x more expensive)

## Upgrading Later

When you're ready to scale:
1. Add usage limits per user
2. Implement rate limiting
3. Consider caching common responses
4. Upgrade to GPT-4 only for premium users

## Current Settings (Optimized for Cost)

✅ Model: `gpt-3.5-turbo` (cost-efficient)
✅ Max tokens: 500 (reduces output costs)
✅ Conversation history: Last 10 messages only
✅ Context loading: Only recent data (last 10-20 entries)

These settings give you great quality while maximizing your $2!

