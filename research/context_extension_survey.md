# Context Extension Techniques Survey

## NTK-Aware RoPE Scaling
- **Principle**: Adjust rotary positional embedding base frequency to preserve neighbourhood geometry as sequence length grows.  
- **Benefits**: Minimal code changes, compatible with most Transformer-based causal LMs.  
- **Considerations**: Requires careful rescaling of attention logits to avoid divergence at very long lengths.

## YaRN (Yet another RoPE eNhancement)
- **Key Insight**: Applies dynamic interpolation during attention computation, enabling 10× token efficiency and 2.5× fewer steps for context adaptation.  
- **Integration Path**: Implement as a drop-in patch around attention modules; pair with progressive curriculum for stability.

## LongRoPE2
- **Improvement**: Reduces perplexity degradation by combining base-frequency scaling with low-rank corrections.  
- **Trade-off**: Slightly higher compute overhead; best suited for target contexts ≥64K.

## Position Interpolation (PI)
- **Usage**: Fine-tune on interpolated positional ids to stretch context length.  
- **Pros**: Simple to implement, works with limited data.  
- **Cons**: Requires additional fine-tuning epochs; may impact short-context accuracy if not mixed properly.

## ALiBi & Hybrid Approaches
- **ALiBi**: Linear biases applied directly to attention scores; inherently unbounded context.  
- **Hybrid**: Combine RoPE scaling with ALiBi offsets for extra stability in extreme contexts (≥128K).

## Recommendations
1. Use NTK-aware RoPE scaling as the default path to 32K.  
2. Offer YaRN as an advanced option for 64K+ contexts.  
3. Provide curriculum fine-tuning hooks to integrate PI when users have extended datasets.  
4. Benchmark hybrid RoPE + ALiBi for future releases once telemetry is in place.
