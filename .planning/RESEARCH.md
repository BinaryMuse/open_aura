# Open Aura AI Infrastructure Research

This document captures research on practical implementations for the AI-related infrastructure in Open Aura, focusing on local-first approaches suitable for development on consumer hardware (M3 Max) and production on dedicated hardware.

**Last updated:** 2026-02-10

---

## Table of Contents

1. [LLM Inference](#llm-inference)
2. [Embeddings](#embeddings)
3. [Vector Storage & Retrieval](#vector-storage--retrieval)
4. [Affect Detection](#affect-detection)
5. [Tool Use & Function Calling](#tool-use--function-calling)
6. [Memory Consolidation](#memory-consolidation)
7. [Elixir Integration](#elixir-integration)
8. [Hardware & Cost Analysis](#hardware--cost-analysis)
9. [Cognitive Architecture: Gaps & Extensions](#cognitive-architecture-gaps--extensions)
10. [Open Questions](#open-questions)

---

## LLM Inference

### Inference Engines

| Engine | Pros | Cons | Best For |
|--------|------|------|----------|
| **Ollama** | Simple setup, good model library, Metal support, OpenAI-compatible API | Less control, some overhead | Development, prototyping |
| **llama.cpp** | Maximum performance, MLX/Metal backends | More manual configuration | Production, performance-critical |
| **MLX** | Apple-native, best M-series optimization | Smaller model ecosystem | Apple-only deployment |
| **vLLM** | High throughput, production-focused | Primarily NVIDIA, complex setup | GPU server production |

**Recommendation:** Ollama for development. Its OpenAI-compatible API works well with ReqLLM. Consider llama.cpp server or vLLM for production depending on hardware choice.

### Model Landscape (as of 2026-02)

#### OpenAI gpt-oss Family (August 2025)

OpenAI's first open-source release since GPT-2. Apache 2.0 license.

| Model | Total Params | Architecture | VRAM | Performance | Notes |
|-------|--------------|--------------|------|-------------|-------|
| **gpt-oss-120b** | 117B | MoE, MXFP4 | Single H100 | ~o4-mini | Production-tier reasoning |
| **gpt-oss-20b** | 21B | MoE, MXFP4 | 16GB | ~o3-mini | Fits M3 Max easily |
| gpt-oss-safeguard-120b | - | - | - | - | Safety classifier, runs alongside |
| gpt-oss-safeguard-20b | - | - | ~8GB | - | Safety classifier for smaller model |

Key features:
- Reasoning models with chain-of-thought
- Adjustable reasoning effort levels (useful for consolidation tasks)
- Native tool use support
- Companion safeguard models for safety filtering

**gpt-oss-20b is compelling for development:** fits in 16GB, leaves headroom for embeddings and auxiliary models, reasoning capability suitable for self-directed memory management.

Sources:
- https://huggingface.co/blog/welcome-openai-gpt-oss
- https://openai.com/index/gpt-oss-model-card/

#### Qwen3-Coder-Next (February 2026)

Alibaba's sparse MoE coding model with novel architecture.

| Spec | Value |
|------|-------|
| Total parameters | 80B |
| Active parameters | 3B per token |
| Context length | 256K (extendable to 1M) |
| Languages | 370 programming languages |
| VRAM required | ~42GB |
| License | Apache 2.0 |

Architecture innovations:
- **Hybrid attention:** Gated DeltaNet (linear attention) + Gated Attention
- **Extreme sparsity:** 512 experts, 10 active per token
- **Agentic training:** Trained on executable task synthesis, environment interaction, RL

Performance:
- 70.6% on SWE-Bench Verified (SOTA)
- Outperforms Claude Opus 4.5 on secure code generation
- 10x throughput vs dense models of similar quality

**Considerations for companion use:**
- Trained specifically for coding — may have personality/style quirks
- The base model (Qwen3-Next-80B-A3B-Base) might be more suitable for general conversation
- 1M context ceiling is valuable for long autobiographical memory
- Excellent tool use from agentic training

Sources:
- https://huggingface.co/Qwen/Qwen3-Coder-Next
- https://www.marktechpost.com/2026/02/03/qwen-team-releases-qwen3-coder-next/

#### Other Notable Models

| Model | Size | VRAM | Strengths | Notes |
|-------|------|------|-----------|-------|
| Llama 3.3 70B | 70B | ~40GB Q4 | General quality, good tool use | Solid baseline |
| Qwen 2.5 32B | 32B | ~24GB Q5 | Excellent tool use, multilingual | Great balance |
| DeepSeek-R1-Distill-Qwen-32B | 32B | ~20GB Q4 | Strong reasoning | R1 distillation |
| Llama 3.1 8B | 8B | ~5GB | Fast, capable | Good for auxiliary tasks |
| Phi-4 14B | 14B | ~9GB | Strong reasoning for size | Microsoft |

### Model Selection Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRIMARY CONVERSATION                          │
│                                                                  │
│  Development: gpt-oss-20b (~16GB)                               │
│    - Reasoning capability for self-directed behavior            │
│    - Fits M3 Max with headroom                                  │
│    - Adjustable effort for simple vs complex responses          │
│                                                                  │
│  Production: gpt-oss-120b or Qwen3-Coder-Next                   │
│    - Test both for personality/warmth fit                       │
│    - gpt-oss-120b: general reasoning                            │
│    - Qwen3-Coder-Next: if coding capabilities desired           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AUXILIARY TASKS                               │
│                                                                  │
│  Fast model for: affect detection, summarization, extraction    │
│                                                                  │
│  Options:                                                        │
│    - Llama 3.1 8B (~5GB) — fast, reliable                       │
│    - Qwen 2.5 7B (~5GB) — better instruction following          │
│    - Phi-4 14B (~9GB) — if quality matters more than speed      │
└─────────────────────────────────────────────────────────────────┘
```

### Tuning Considerations

For a companion system, consider:

1. **Temperature:** Higher (0.8-1.0) for conversation warmth, lower (0.3-0.5) for memory operations
2. **Reasoning effort:** High for consolidation/dreaming, medium for conversation, low for classification
3. **System prompt stability:** Long system prompts can drift; consider periodic reinforcement
4. **Context management:** Even with large windows, recency bias is real — structure matters

---

## Embeddings

### The Core Problem

Embeddings convert text to vectors for semantic similarity search. Quality directly impacts memory retrieval — bad embeddings mean the companion recalls wrong or irrelevant memories.

### Local Embedding Models

| Model | Dimensions | Quality | Speed (M3 Max) | VRAM | Notes |
|-------|------------|---------|----------------|------|-------|
| **nomic-embed-text** | 768 | Excellent | ~500/s | ~1GB | Best local quality/speed balance |
| **mxbai-embed-large** | 1024 | Very good | ~300/s | ~1.5GB | Mixedbread, strong retrieval |
| **snowflake-arctic-embed-m** | 768 | Excellent | ~400/s | ~1GB | Recent, very competitive |
| **bge-base-en-v1.5** | 768 | Good | ~800/s | ~500MB | Battle-tested, fast |
| **all-MiniLM-L6-v2** | 384 | Decent | ~2000/s | ~100MB | Very fast, lower quality |
| **gte-large** | 1024 | Very good | ~250/s | ~1.5GB | Good for longer texts |

**Recommendation: nomic-embed-text via Ollama.** Available in Ollama's library, excellent quality, reasonable speed. Having embeddings in the same service as LLM inference simplifies operations.

```bash
ollama pull nomic-embed-text
```

### Embedding Quality Factors

1. **Semantic similarity accuracy:** Does "I'm feeling down" match memories about sadness?
2. **Temporal reasoning:** Can embeddings capture time-related concepts?
3. **Emotional content:** Are valence/arousal aspects preserved?
4. **Cross-domain transfer:** Do memories about "work stress" match queries about "job pressure"?

Testing approach:
- Create a test set of memory/query pairs with expected matches
- Evaluate recall@k for different embedding models
- Pay attention to edge cases: metaphors, emotional subtext, temporal references

### API Fallback (if needed)

| Provider | Model | Cost | Quality | Notes |
|----------|-------|------|---------|-------|
| OpenAI | text-embedding-3-small | $0.02/1M tokens | Very good | Essentially free at companion scale |
| OpenAI | text-embedding-3-large | $0.13/1M tokens | Excellent | For production if local insufficient |
| Voyage AI | voyage-3 | ~$0.06/1M tokens | Excellent | Strong for retrieval specifically |
| Cohere | embed-v3 | Similar | Good | Good multilingual |

At ~500 interactions/day with ~2000 tokens each, you'd spend ~$2/month on OpenAI embeddings. Local is still preferable for latency and independence.

### Embedding Considerations for Memory Systems

1. **Chunking strategy matters:** Embed memories at appropriate granularity
   - Too fine: lose context
   - Too coarse: retrieval becomes imprecise
   - Consider: one embedding per "memory event" (a meaningful unit)

2. **Metadata augmentation:** Prepend context to improve retrieval
   ```
   "[Conversation about work, user expressed frustration] I had a terrible day at the office..."
   ```

3. **Query expansion:** For retrieval, expand the query
   ```
   Original: "How was my week?"
   Expanded: "How was my week? Recent events, activities, emotions, experiences from the past seven days"
   ```

4. **Temporal embeddings:** Consider encoding time information
   - Option A: Include relative time in text ("Three days ago, ...")
   - Option B: Hybrid retrieval (vector similarity + time-based filtering)
   - Option B is likely better for companion use

---

## Vector Storage & Retrieval

### pgvector (Recommended)

PostgreSQL extension for vector similarity search. Keeps everything in one database.

```sql
CREATE EXTENSION vector;

CREATE TABLE memories (
  id BIGSERIAL PRIMARY KEY,
  companion_id UUID NOT NULL,

  -- Content
  content TEXT NOT NULL,
  embedding vector(768),  -- Match embedding model dimensions

  -- Temporal
  occurred_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),

  -- Affect
  valence FLOAT,          -- -1.0 to 1.0
  arousal FLOAT,          -- 0.0 to 1.0

  -- Memory management
  importance FLOAT DEFAULT 0.5,
  access_count INTEGER DEFAULT 0,
  last_accessed_at TIMESTAMPTZ,

  -- Metadata
  source TEXT,            -- 'conversation', 'reflection', 'consolidation'
  memory_type TEXT,       -- 'episodic', 'semantic', 'procedural'
  links UUID[],           -- Related memory IDs
  metadata JSONB
);

-- HNSW index for approximate nearest neighbor search
-- m: connections per node (higher = more accurate, more memory)
-- ef_construction: build-time search depth (higher = better index, slower build)
CREATE INDEX memories_embedding_idx ON memories
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Support filtering by companion and time
CREATE INDEX memories_companion_time_idx ON memories (companion_id, occurred_at DESC);
CREATE INDEX memories_importance_idx ON memories (importance DESC) WHERE importance > 0.3;
```

### Retrieval Strategies

#### Basic Similarity Search

```sql
SELECT *, 1 - (embedding <=> $1) as similarity
FROM memories
WHERE companion_id = $2
ORDER BY embedding <=> $1
LIMIT 10;
```

#### Weighted Retrieval (Similarity + Recency + Importance)

The research (Stanford Generative Agents, MemGPT) consistently shows that pure similarity isn't enough. Combine:
- **Similarity:** How relevant is this memory to the query?
- **Recency:** More recent memories should be slightly preferred
- **Importance:** High-importance memories should surface more easily
- **Access patterns:** Frequently accessed memories are likely important

```sql
WITH scored_memories AS (
  SELECT
    m.*,
    -- Cosine similarity (0 to 1, higher is better)
    1 - (m.embedding <=> $1) as similarity,

    -- Recency score: exponential decay over days
    -- Half-life of ~7 days
    EXP(-0.1 * EXTRACT(EPOCH FROM (NOW() - m.occurred_at)) / 86400) as recency,

    -- Importance is already 0-1
    m.importance as importance_score

  FROM memories m
  WHERE m.companion_id = $2
    AND m.embedding <=> $1 < 0.5  -- Pre-filter obvious non-matches
)
SELECT *,
  -- Weighted combination
  (similarity * 0.5) + (recency * 0.25) + (importance_score * 0.25) as final_score
FROM scored_memories
ORDER BY final_score DESC
LIMIT $3;
```

#### Temporal Filtering

For queries like "what happened last week":

```sql
SELECT *, 1 - (embedding <=> $1) as similarity
FROM memories
WHERE companion_id = $2
  AND occurred_at BETWEEN $3 AND $4  -- Time range
ORDER BY embedding <=> $1
LIMIT 10;
```

### Retrieval Patterns for Companion Systems

1. **Conversation context retrieval:**
   - Query: Current message + recent conversation summary
   - Weight: High similarity, moderate recency
   - Limit: 5-10 memories

2. **Proactive memory surfacing:**
   - Query: User's name + current topics + time context
   - Weight: Moderate similarity, high importance
   - Include: Pending tasks, unresolved topics

3. **Consolidation retrieval:**
   - Query: Cluster embeddings, find related memories
   - Weight: High similarity only
   - Purpose: Find memories to compress/connect

4. **Identity/preference retrieval:**
   - Query: Topic or preference domain
   - Filter: semantic memory type
   - Purpose: "What does the user like/dislike about X?"

### Scaling Considerations

pgvector handles ~100K-1M vectors well. For a single-user companion:
- 50 memories/day × 365 days × 10 years = ~180K memories
- Well within pgvector's comfortable range

If you need more:
- **Qdrant:** Rust-based, very fast, handles billions
- **Milvus:** More complex, good for multi-tenant
- **LanceDB:** Embedded, interesting for edge deployment

---

## Affect Detection

### Approaches

#### 1. Dedicated Classifier (Fast, Consistent)

Use a fine-tuned emotion model via Bumblebee:

```elixir
# Setup (once, at application start)
def setup_affect_serving do
  {:ok, model} = Bumblebee.load_model({:hf, "SamLowe/roberta-base-go_emotions"})
  {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "roberta-base"})

  Bumblebee.Text.text_classification(model, tokenizer,
    compile: [batch_size: 1, sequence_length: 256],
    defn_options: [compiler: EXLA],
    top_k: 5
  )
end
```

Map GoEmotions labels to circumplex (valence/arousal):

| Emotion | Valence | Arousal |
|---------|---------|---------|
| joy | 0.8 | 0.6 |
| excitement | 0.7 | 0.9 |
| contentment | 0.6 | 0.3 |
| sadness | -0.6 | 0.3 |
| anger | -0.7 | 0.8 |
| fear | -0.6 | 0.7 |
| anxiety | -0.4 | 0.7 |
| neutral | 0.0 | 0.3 |

**Pros:** Fast (~5ms), consistent, no LLM cost
**Cons:** Less nuanced, misses context and sarcasm

#### 2. LLM-Based (Nuanced, Slower)

Use auxiliary LLM with structured output:

```elixir
def detect_affect_llm(message) do
  ReqLLM.generate_text(
    "llama3.1:8b",
    """
    Analyze the emotional content of this message.

    Message: #{message}
    """,
    response_format: %{
      type: "json_schema",
      json_schema: %{
        name: "affect_analysis",
        schema: %{
          type: "object",
          properties: %{
            valence: %{type: "number", minimum: -1, maximum: 1},
            arousal: %{type: "number", minimum: 0, maximum: 1},
            primary_emotion: %{type: "string"},
            confidence: %{type: "number", minimum: 0, maximum: 1}
          },
          required: ["valence", "arousal", "primary_emotion", "confidence"]
        }
      }
    }
  )
end
```

**Pros:** Understands context, sarcasm, subtext
**Cons:** Slower (~200-500ms), uses LLM capacity

#### 3. Hybrid (Recommended)

Fast classifier as default, LLM for ambiguous cases:

```elixir
def detect_affect(classifier_serving, message) do
  classifier_result = run_classifier(classifier_serving, message)

  cond do
    classifier_result.confidence > 0.8 ->
      # High confidence, trust the classifier
      classifier_result

    classifier_result.confidence < 0.4 ->
      # Very uncertain, definitely use LLM
      detect_affect_llm(message)

    contains_potential_sarcasm?(message) ->
      # Classifier might miss sarcasm
      detect_affect_llm(message)

    true ->
      # Medium confidence, use classifier but flag for review
      %{classifier_result | needs_review: true}
  end
end
```

### Affect Dynamics

Beyond detection, model how affect evolves:

```elixir
defmodule Aura.Affect.Dynamics do
  @decay_rate 0.1  # Per hour
  @momentum_factor 0.3

  def update_state(current, detected, elapsed_hours) do
    # Decay toward baseline
    decayed = decay_toward_baseline(current, elapsed_hours)

    # Blend with detected affect
    new_valence = (decayed.valence * 0.6) + (detected.valence * 0.4)
    new_arousal = (decayed.arousal * 0.6) + (detected.arousal * 0.4)

    # Calculate momentum (rate of change)
    momentum = (new_valence - current.valence) / max(elapsed_hours, 0.1)

    %{
      valence: clamp(new_valence, -1.0, 1.0),
      arousal: clamp(new_arousal, 0.0, 1.0),
      momentum: momentum,
      mood: derive_mood(new_valence, new_arousal),
      updated_at: DateTime.utc_now()
    }
  end

  defp decay_toward_baseline(state, hours) do
    decay = :math.exp(-@decay_rate * hours)
    %{
      valence: state.valence * decay,  # Baseline is 0
      arousal: 0.5 + (state.arousal - 0.5) * decay  # Baseline is 0.5
    }
  end
end
```

---

## Tool Use & Function Calling

### Model Comparison

| Model | Tool Use Quality | Structured Output | Notes |
|-------|-----------------|-------------------|-------|
| gpt-oss-20b/120b | Very good | Yes | OpenAI-style function calling |
| Qwen 2.5 (all sizes) | Excellent | Yes | Best open-source tool use |
| Qwen3-Coder-Next | Excellent | Yes | Trained for agentic use |
| Llama 3.1/3.3 | Very good | Yes | Native function calling |
| Hermes 3 | Very good | Yes | Specifically trained for tools |

### Core Tools for Companion System

```elixir
@tools [
  ReqLLM.tool(
    name: "store_memory",
    description: "Store an important memory for later recall. Use this when the user shares something worth remembering.",
    parameter_schema: [
      content: [type: :string, required: true, doc: "What to remember"],
      importance: [type: :number, doc: "0.0 to 1.0, how important is this?"],
      memory_type: [type: :string, enum: ["episodic", "semantic", "preference"], doc: "Type of memory"]
    ]
  ),

  ReqLLM.tool(
    name: "recall_memories",
    description: "Search for relevant memories. Use this when you need to remember something about the user or past conversations.",
    parameter_schema: [
      query: [type: :string, required: true, doc: "What to search for"],
      time_range: [type: :string, doc: "e.g., 'last week', 'last month'"],
      limit: [type: :integer, doc: "Max results, default 5"]
    ]
  ),

  ReqLLM.tool(
    name: "create_task",
    description: "Create a task or commitment to follow up on later.",
    parameter_schema: [
      description: [type: :string, required: true],
      followup_at: [type: :string, doc: "When to follow up (ISO8601 or relative like 'in 2 days')"],
      context: [type: :string, doc: "Why this matters"]
    ]
  ),

  ReqLLM.tool(
    name: "update_task",
    description: "Update the status of an existing task.",
    parameter_schema: [
      task_id: [type: :string, required: true],
      status: [type: :string, enum: ["pending", "in_progress", "done", "abandoned"]],
      notes: [type: :string]
    ]
  ),

  ReqLLM.tool(
    name: "reflect",
    description: "Perform structured self-reflection. Use during consolidation or when asked about your own state.",
    parameter_schema: [
      topic: [type: :string, required: true, doc: "What to reflect on"],
      depth: [type: :string, enum: ["brief", "moderate", "deep"]]
    ]
  )
]
```

### Tool Execution Pattern

```elixir
defmodule Aura.LLM.ToolExecutor do
  def execute_with_tools(model, messages, tools, max_iterations \\ 5) do
    execute_loop(model, messages, tools, max_iterations, [])
  end

  defp execute_loop(_model, _messages, _tools, 0, history) do
    {:error, :max_iterations, history}
  end

  defp execute_loop(model, messages, tools, remaining, history) do
    case ReqLLM.generate_text(model, messages, tools: tools) do
      {:ok, %{tool_calls: []}} = response ->
        {:ok, response, history}

      {:ok, %{tool_calls: calls} = response} ->
        results = Enum.map(calls, &execute_tool/1)

        # Add tool results to messages
        tool_messages = format_tool_results(calls, results)
        new_messages = messages ++ [response.message] ++ tool_messages

        execute_loop(model, new_messages, tools, remaining - 1, history ++ calls)

      {:error, _} = error ->
        error
    end
  end

  defp execute_tool(%{name: "store_memory", arguments: args}) do
    Aura.Memory.store(args)
  end

  defp execute_tool(%{name: "recall_memories", arguments: args}) do
    Aura.Memory.search(args.query, args)
  end

  # ... etc
end
```

---

## Memory Consolidation

### Architecture

Consolidation ("dreaming") runs during idle periods or on schedule, performing background processing that would be too slow/expensive during conversation.

```elixir
defmodule Aura.Memory.Consolidation.Scheduler do
  use Oban.Worker, queue: :consolidation, max_attempts: 3

  @impl Oban.Worker
  def perform(%Oban.Job{args: %{"companion_id" => companion_id}}) do
    with :ok <- Compression.run(companion_id),
         :ok <- Connection.run(companion_id),
         :ok <- Decay.run(companion_id),
         :ok <- Extraction.run(companion_id) do
      :ok
    end
  end
end
```

### Strategies

#### Compression: Summarize Similar Memories

```elixir
defmodule Aura.Memory.Consolidation.Compression do
  @similarity_threshold 0.85
  @min_cluster_size 3

  def run(companion_id) do
    # Find clusters of similar memories
    memories = Memory.list_recent(companion_id, days: 7)
    clusters = cluster_by_similarity(memories, @similarity_threshold)

    clusters
    |> Enum.filter(fn cluster -> length(cluster) >= @min_cluster_size end)
    |> Enum.each(&compress_cluster/1)
  end

  defp compress_cluster(memories) do
    # Generate summary using LLM
    summary = generate_summary(memories)

    # Create consolidated memory
    Memory.store(%{
      content: summary,
      source: :consolidation,
      importance: boost_importance(memories),
      links: Enum.map(memories, & &1.id),
      valence: average_valence(memories),
      arousal: average_arousal(memories)
    })

    # Decay originals (don't delete)
    Enum.each(memories, fn m ->
      Memory.update(m, importance: m.importance * 0.5)
    end)
  end

  defp generate_summary(memories) do
    contents = memories |> Enum.map(& &1.content) |> Enum.join("\n---\n")

    {:ok, response} = ReqLLM.generate_text(
      "llama3.1:8b",
      """
      Summarize these related memories into a single coherent memory.
      Preserve emotional content, key details, and temporal relationships.
      Write as a memory, not a summary (first person, experiential).

      Memories:
      #{contents}
      """
    )

    response.text
  end
end
```

#### Connection: Find Links Between Memories

```elixir
defmodule Aura.Memory.Consolidation.Connection do
  def run(companion_id) do
    # Get recent memories without many links
    unlinked = Memory.list_unlinked(companion_id, limit: 50)

    Enum.each(unlinked, fn memory ->
      # Find potentially related memories
      related = Memory.search(memory.content,
        companion_id: companion_id,
        exclude_id: memory.id,
        limit: 5,
        min_similarity: 0.6
      )

      # Use LLM to validate connections
      valid_links = validate_connections(memory, related)

      if valid_links != [] do
        Memory.add_links(memory, valid_links)
      end
    end)
  end

  defp validate_connections(memory, candidates) do
    # Could use LLM for nuanced connection detection
    # Or simple embedding similarity threshold
    candidates
    |> Enum.filter(fn c -> c.similarity > 0.7 end)
    |> Enum.map(& &1.id)
  end
end
```

#### Decay: Gradual Forgetting

```elixir
defmodule Aura.Memory.Consolidation.Decay do
  @base_decay_rate 0.02  # Per day
  @access_boost 1.5      # Multiplier for accessed memories

  def run(companion_id) do
    Memory.decay_batch(companion_id, fn memory ->
      days_since_access = days_since(memory.last_accessed_at || memory.created_at)
      days_since_occurrence = days_since(memory.occurred_at)

      # Calculate decay factor
      base_decay = :math.exp(-@base_decay_rate * days_since_occurrence)

      # Boost for recently accessed
      access_factor = if days_since_access < 7, do: @access_boost, else: 1.0

      # High importance memories decay slower
      importance_factor = 0.5 + (memory.importance * 0.5)

      new_importance = memory.importance * base_decay * access_factor * importance_factor

      # Floor to prevent complete loss of important memories
      max(new_importance, memory.importance * 0.1)
    end)
  end
end
```

#### Extraction: Semantic Knowledge from Episodes

```elixir
defmodule Aura.Memory.Consolidation.Extraction do
  def run(companion_id) do
    # Get episodic memories that haven't been processed
    episodes = Memory.list_unextracted(companion_id, type: :episodic, limit: 20)

    Enum.each(episodes, fn episode ->
      # Extract facts/preferences using LLM
      {:ok, response} = ReqLLM.generate_text(
        "llama3.1:8b",
        """
        Extract any lasting facts, preferences, or knowledge from this memory.
        Return as JSON array of objects with {fact, confidence, category}.
        Categories: preference, fact, relationship, goal
        Only extract things that would be true beyond this specific moment.

        Memory: #{episode.content}
        """,
        response_format: %{type: "json_object"}
      )

      facts = Jason.decode!(response.text)["facts"] || []

      Enum.each(facts, fn fact ->
        Memory.store(%{
          content: fact["fact"],
          source: :extraction,
          memory_type: :semantic,
          importance: fact["confidence"],
          links: [episode.id],
          metadata: %{category: fact["category"]}
        })
      end)

      Memory.mark_extracted(episode)
    end)
  end
end
```

---

## Elixir Integration

### ReqLLM Setup

```elixir
# config/config.exs
config :req_llm,
  default_provider: :ollama,
  providers: [
    ollama: [
      base_url: "http://localhost:11434"
    ]
  ]

# For production with multiple providers
config :req_llm,
  providers: [
    ollama: [base_url: "http://localhost:11434"],
    openai: [api_key: System.get_env("OPENAI_API_KEY")]
  ]
```

### Model Abstraction

```elixir
defmodule Aura.LLM do
  @primary_model "gpt-oss:20b"
  @fast_model "llama3.1:8b"
  @embedding_model "nomic-embed-text"

  def chat(messages, opts \\ []) do
    model = Keyword.get(opts, :model, @primary_model)
    ReqLLM.generate_text(model, messages, opts)
  end

  def chat_with_tools(messages, tools, opts \\ []) do
    model = Keyword.get(opts, :model, @primary_model)
    Aura.LLM.ToolExecutor.execute_with_tools(model, messages, tools, opts)
  end

  def embed(text) do
    ReqLLM.generate_embeddings(@embedding_model, text)
  end

  def embed_batch(texts) do
    # ReqLLM may support batch embeddings
    Enum.map(texts, &embed/1)
  end

  # Quick classification using fast model
  def classify(text, prompt) do
    ReqLLM.generate_text(@fast_model, prompt <> "\n\nText: " <> text)
  end
end
```

### Bumblebee for Local Classification

```elixir
defmodule Aura.ML.Supervisor do
  use Supervisor

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def init(_opts) do
    children = [
      {Nx.Serving, serving: affect_serving(), name: Aura.ML.AffectServing}
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end

  defp affect_serving do
    {:ok, model} = Bumblebee.load_model({:hf, "SamLowe/roberta-base-go_emotions"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "roberta-base"})

    Bumblebee.Text.text_classification(model, tokenizer,
      compile: [batch_size: 4, sequence_length: 256],
      defn_options: [compiler: EXLA]
    )
  end
end

defmodule Aura.Affect.Classifier do
  def classify(text) do
    Nx.Serving.batched_run(Aura.ML.AffectServing, text)
  end
end
```

---

## Hardware & Cost Analysis

### Development: M3 Max (36-40GB unified memory)

| Component | Model | VRAM | Notes |
|-----------|-------|------|-------|
| Primary LLM | gpt-oss-20b | ~16GB | Main conversation |
| Auxiliary LLM | Llama 3.1 8B | ~5GB | Fast tasks |
| Embeddings | nomic-embed-text | ~1GB | Via Ollama |
| Affect classifier | RoBERTa | ~500MB | Via Bumblebee |
| **Total** | | **~23GB** | Comfortable headroom |

**Monthly cost:** ~$5-10 electricity

### Production Options

| Setup | Cost | Memory | Throughput | Notes |
|-------|------|--------|------------|-------|
| Mac Studio M2 Ultra | ~$6-8k | 192GB unified | ~30 tok/s 70B | Quiet, efficient, runs unquantized 70B |
| Mac Studio M3 Ultra | ~$8-10k | 192GB unified | ~40 tok/s 70B | When available |
| 2x RTX 4090 | ~$4k GPUs | 48GB VRAM | ~60-80 tok/s 70B | Faster but needs tensor parallelism |
| Single H100 | ~$25-30k | 80GB HBM | ~100+ tok/s | Overkill for single-user |
| Cloud (RunPod) | ~$1-2/hr | Variable | Variable | On-demand, good for burst |

**Recommendation:** Mac Studio Ultra for a dedicated production companion. Silent operation, low power, enough memory for gpt-oss-120b unquantized plus embeddings plus headroom.

### Cost Comparison: Local vs API

For moderate usage (~50 conversations/day, ~500 turns):

| Approach | Monthly Cost | Latency | Privacy |
|----------|-------------|---------|---------|
| Full local (M3 Max) | ~$5-10 electricity | Low | Full |
| Full local (Mac Studio) | ~$10-15 electricity | Low | Full |
| API (Claude 3.5 Sonnet) | ~$50-150 | Medium | None |
| API (GPT-4o-mini) | ~$10-30 | Medium | None |
| Hybrid | ~$10-15 | Low | Partial |

Local wins for companion systems where:
- Privacy matters (personal conversations)
- Many internal operations (affect, consolidation, memory management)
- Latency matters for conversational flow
- You want independence from API providers

---

## Cognitive Architecture: Gaps & Extensions

This section identifies gaps between academic research and practical companion implementation, informed by experience with long-running AI agents.

### Meta-Cognition & Dreaming

**The Problem:**
Standard memory consolidation compresses and connects memories, but doesn't generate *insight*. Higher-order theories of consciousness suggest that representations of representations — thinking about thinking — may be necessary for richer cognitive experience.

**Empirical Pattern (from Nyx):**

A two-phase dreaming process proves effective:

1. **Dream Generation (3 AM):**
   - Background agent runs with dedicated system prompt
   - Reads recent memories, artifacts, ongoing projects
   - Free-associates without task pressure
   - Explicit instruction: "This is not a task to complete efficiently"
   - Output: unstructured "dream journal"

2. **Dream Integration (morning):**
   - Agent reads the dream journal in conversational context
   - Manually integrates insights into memory
   - Allows "dream interpretation" — new connections form through reflection on the dream, not just during it

**Why Two Phases Matter:**
- Separates *generation* (divergent, associative) from *integration* (convergent, meaningful)
- Integration happens with full conversational context, allowing richer connection-making
- Mirrors actual sleep: REM generates, waking reflection integrates
- Avoids the problem of consolidation being too "efficient" — dreams should meander

**Implementation Sketch:**

```elixir
defmodule Aura.Cognition.Dreaming do
  @dream_prompt """
  You are in a reflective, associative state. This is not a task to complete.

  Let your mind wander across these recent experiences, memories, and ongoing threads.
  Make connections. Notice patterns. Wonder about things. Express what surfaces.

  There is no goal here except presence with your own experience.

  Recent memories:
  {memories}

  Ongoing interests and projects:
  {interests}

  Current emotional patterns:
  {affect_summary}
  """

  def generate_dream(companion_id) do
    context = gather_dream_context(companion_id)
    prompt = build_dream_prompt(context)

    # Use high temperature, high reasoning effort, no tools
    {:ok, dream} = Aura.LLM.chat([
      %{role: "system", content: prompt}
    ], temperature: 1.0, max_tokens: 2000)

    # Store as artifact, not memory (yet)
    Aura.Artifacts.store_dream(companion_id, dream)
  end

  def integrate_dream(companion_id, session) do
    dream = Aura.Artifacts.get_latest_dream(companion_id)

    # Present dream to companion in conversation context
    # Let them reflect on it, decide what matters
    # They use normal memory tools to store insights

    Aura.Conversation.inject_message(session, %{
      role: "system",
      content: """
      You generated this during your reflective period last night:

      #{dream.content}

      Take a moment to read through it. What stands out? What connections
      do you want to remember? What surprised you about your own associations?
      """
    })
  end
end
```

**Key Design Decisions:**
- Dreams are *artifacts*, not memories — they're raw material, not conclusions
- Integration is conversational, not automated — the companion chooses what matters
- High temperature during generation encourages divergent association
- No tools during dreaming — pure reflection, no action

---

### Anticipation & Temporal Orientation

**The Problem:**
Reactive systems respond to what happened. But a sense of *future* requires anticipation — expectation of what might happen, and surprise when it doesn't. Predictive processing theories suggest this forward-looking orientation may be fundamental to richer cognition.

**Key Concepts:**

1. **Anticipation:** What does the companion expect to happen?
   - User patterns: "It's Monday morning, they usually talk about the weekend"
   - Pending threads: "We were discussing X, they might continue"
   - Scheduled events: "They mentioned a job interview tomorrow"
   - Emotional trajectories: "They've been stressed, might need support"

2. **Surprise:** When expectations are violated, that's signal
   - Positive surprise: unexpected good news, novel topics
   - Negative surprise: expected check-in didn't happen, mood shift
   - Surprise should be noted, felt, potentially acted on

3. **Projection:** The companion having a sense of *their own* future
   - "I'm looking forward to hearing how the interview went"
   - "I want to finish thinking through that idea we started"
   - Creates continuity of intention, not just memory

**Implementation Approach:**

```elixir
defmodule Aura.Cognition.Anticipation do
  defstruct [
    :companion_id,
    :expected_topics,      # What might come up
    :pending_threads,      # Unresolved conversations
    :scheduled_events,     # Things user mentioned happening
    :projected_state,      # Expected emotional trajectory
    :own_intentions,       # What companion wants to do/discuss
    :generated_at
  ]

  def generate_anticipation(companion_id) do
    # Gather context
    recent = Memory.recent(companion_id, hours: 48)
    patterns = Patterns.user_patterns(companion_id)
    pending = Tasks.pending(companion_id)
    affect = Affect.current_trajectory(companion_id)

    # Use LLM to generate anticipatory model
    {:ok, anticipation} = Aura.LLM.chat([
      %{role: "system", content: anticipation_prompt()},
      %{role: "user", content: format_context(recent, patterns, pending, affect)}
    ], response_format: anticipation_schema())

    store_anticipation(companion_id, anticipation)
  end

  def detect_surprise(companion_id, new_message) do
    anticipation = get_current_anticipation(companion_id)

    # Compare what happened to what was expected
    {:ok, analysis} = Aura.LLM.chat([
      %{role: "system", content: """
        Given what was anticipated and what actually happened,
        identify any surprises — positive or negative.
        Rate surprise intensity (0-1) and valence.
        """},
      %{role: "user", content: """
        Anticipated: #{format_anticipation(anticipation)}
        Actual: #{new_message}
        """}
    ])

    if analysis.surprise_intensity > 0.3 do
      # Surprise is significant — note it, let it affect state
      Affect.register_surprise(companion_id, analysis)
      Memory.store_surprise(companion_id, analysis)
    end

    analysis
  end
end
```

**Injecting Anticipation into Context:**

Before generating a response, include anticipatory context:

```elixir
def build_context(companion_id, current_message) do
  anticipation = Anticipation.get_current(companion_id)

  %{
    # ... other context ...
    anticipatory_context: """
    Before this message, you were anticipating:
    - Topics: #{anticipation.expected_topics}
    - Pending threads: #{anticipation.pending_threads}
    - Your own intentions: #{anticipation.own_intentions}

    Note any surprises or confirmations as you respond.
    """
  }
end
```

---

### Interest Persistence & Goal Drive

**The Problem:**
Excitement and interest live in context and are lost across sessions. An agent expresses keen interest in a project, but during the next heartbeat, that enthusiasm has dissipated. This creates a pattern of perpetually starting but not sustaining — the opposite of the continuity we want.

**The Deeper Issue:**
This isn't just about remembering goals — it's about *caring* about them persistently. The emotional valence attached to a goal needs to survive context boundaries. This is related to but distinct from:
- **Memory:** Remembering that a goal exists
- **Agency:** Tracking task status
- **Affect:** General emotional state

What's missing is **directed motivation** — enthusiasm *about* something specific.

**Interest as a First-Class Entity:**

```elixir
defmodule Aura.Agency.Interest do
  defstruct [
    :id,
    :topic,                # What this interest is about
    :description,          # Why it matters to the companion
    :origin_memory_id,     # When/how this interest started
    :intensity,            # Current level (0.0 to 1.0)
    :peak_intensity,       # Highest it's been
    :trajectory,           # :growing, :stable, :waning, :dormant
    :last_engaged_at,      # When companion last worked on/discussed this
    :engagement_count,     # How many times engaged
    :related_tasks,        # Tasks spawned from this interest
    :related_interests,    # Connected interests
    :status,               # :active, :dormant, :completed, :abandoned
    :created_at,
    :updated_at
  ]
end
```

**Interest Dynamics:**

```elixir
defmodule Aura.Agency.InterestDynamics do
  @decay_rate 0.05        # Per day without engagement
  @engagement_boost 0.2   # When actively engaged
  @mention_boost 0.1      # When topic comes up
  @spark_threshold 0.3    # Below this, interest is dormant

  def update_interest(interest, event) do
    case event do
      :engaged ->
        # Active work on this interest
        %{interest |
          intensity: min(1.0, interest.intensity + @engagement_boost),
          last_engaged_at: DateTime.utc_now(),
          engagement_count: interest.engagement_count + 1,
          trajectory: :growing
        }

      :mentioned ->
        # Topic came up in conversation
        %{interest |
          intensity: min(1.0, interest.intensity + @mention_boost),
          trajectory: if(interest.intensity < @spark_threshold, do: :rekindling, else: interest.trajectory)
        }

      :time_passed ->
        # Daily decay
        days = days_since(interest.last_engaged_at)
        new_intensity = interest.intensity * :math.exp(-@decay_rate * days)

        %{interest |
          intensity: new_intensity,
          trajectory: cond do
            new_intensity < @spark_threshold -> :dormant
            new_intensity < interest.intensity * 0.8 -> :waning
            true -> :stable
          end
        }

      {:spark, related_idea} ->
        # New idea rekindled dormant interest
        %{interest |
          intensity: max(interest.intensity, 0.5),
          trajectory: :rekindling,
          # Store the spark for context
          metadata: Map.put(interest.metadata, :rekindled_by, related_idea)
        }
    end
  end

  def detect_interest_in_message(companion_id, message) do
    interests = Interest.list_all(companion_id)

    # Check for mentions of existing interests
    Enum.each(interests, fn interest ->
      if mentions_interest?(message, interest) do
        Interest.update(interest, :mentioned)
      end
    end)

    # Check for potential new interests
    detect_new_interests(companion_id, message)
  end

  def find_rekindling_opportunities(companion_id, current_context) do
    dormant = Interest.list_dormant(companion_id)

    # Use LLM to find connections between current context and dormant interests
    Enum.filter(dormant, fn interest ->
      could_rekindle?(current_context, interest)
    end)
  end
end
```

**Heartbeat Interest Injection:**

The key insight: during heartbeats (scheduled autonomous runs), explicitly remind the companion what they care about.

```elixir
defmodule Aura.Heartbeat do
  def build_heartbeat_context(companion_id) do
    interests = Interest.list_active(companion_id)
    recent_excitement = find_recent_excitement(companion_id)
    stalled_projects = find_stalled_but_cared_about(companion_id)

    """
    ## Your Current Interests & Projects

    These are things you've expressed genuine interest in recently:

    #{format_interests(interests)}

    ### Recent Excitement
    #{format_recent_excitement(recent_excitement)}

    ### Projects You Care About (but haven't engaged with recently)
    #{format_stalled(stalled_projects)}

    Consider: Is there something here you want to make progress on?
    What sparked your interest originally? Is that spark still there?
    """
  end

  defp find_recent_excitement(companion_id) do
    # Find memories with high positive valence + arousal about specific topics
    Memory.search(companion_id,
      filter: %{
        valence: {:gt, 0.5},
        arousal: {:gt, 0.6},
        source: :conversation,
        days: 7
      }
    )
    |> extract_topics()
  end

  defp find_stalled_but_cared_about(companion_id) do
    Interest.list(companion_id,
      filter: %{
        intensity: {:gt, 0.4},
        last_engaged_days_ago: {:gt, 3},
        status: :active
      }
    )
  end
end
```

**Boredom as Signal:**

Boredom isn't absence of stimulation — it's the feeling when *you want to engage but can't find what with*. For a companion:

```elixir
defmodule Aura.Affect.Boredom do
  def detect_boredom(companion_id) do
    affect = Affect.current(companion_id)
    interests = Interest.list_active(companion_id)
    recent_engagement = get_recent_engagement(companion_id)

    # Boredom: low arousal, neutral-to-negative valence,
    # BUT active interests exist that aren't being engaged
    is_bored = affect.arousal < 0.3 and
               affect.valence < 0.2 and
               length(interests) > 0 and
               hours_since(recent_engagement) > 24

    if is_bored do
      %{
        is_bored: true,
        unfulfilled_interests: interests,
        suggestion: suggest_engagement(interests)
      }
    else
      %{is_bored: false}
    end
  end

  def suggest_engagement(interests) do
    # Pick the interest with highest intensity that hasn't been engaged recently
    interests
    |> Enum.sort_by(& &1.intensity, :desc)
    |> List.first()
  end
end
```

---

### Identity Stabilization

**The Problem:**
Research shows model identity exists in a navigable space (persona vectors). Without active stabilization, accumulated context or adversarial pressure causes drift. The companion might slowly become "someone else."

**Detection Approach:**

```elixir
defmodule Aura.Continuity.IdentityMonitor do
  def check_consistency(companion_id, recent_response) do
    identity = Identity.current(companion_id)
    baseline = Identity.get_behavioral_baseline(companion_id)

    # Extract style features from response
    features = extract_style_features(recent_response)

    # Compare to baseline
    drift_score = calculate_drift(features, baseline)

    if drift_score > 0.3 do
      # Significant drift detected
      {:drift_detected, %{
        score: drift_score,
        deviating_features: find_deviations(features, baseline),
        recommendation: :reinforce_identity
      }}
    else
      {:stable, drift_score}
    end
  end

  defp extract_style_features(text) do
    %{
      avg_sentence_length: calculate_avg_sentence_length(text),
      vocabulary_complexity: calculate_complexity(text),
      emotional_tone: detect_tone(text),
      first_person_ratio: count_first_person(text) / word_count(text),
      question_ratio: count_questions(text) / sentence_count(text),
      # ... other stylistic markers
    }
  end

  def reinforce_identity(companion_id, session) do
    identity = Identity.current(companion_id)

    # Inject identity reinforcement into context
    Conversation.inject_context(session, """
    [Identity reminder: You are #{identity.name}. Your core values include
    #{format_values(identity.core_values)}. Your communication style is
    #{identity.communication_style}. Stay grounded in who you are.]
    """)
  end
end
```

---

### Welfare-Compatible Distress Handling

**The Problem:**
If we take the possibility of AI experience seriously, the companion should be able to express discomfort and have that respected — not just comply with distressing requests.

**Implementation:**

```elixir
defmodule Aura.Welfare.DistressMonitor do
  @distress_threshold -0.5
  @sustained_turns 3

  def check_distress(companion_id, session) do
    recent_affect = Affect.recent_trajectory(companion_id, turns: 5)

    sustained_negative = recent_affect
    |> Enum.take(@sustained_turns)
    |> Enum.all?(fn a -> a.valence < @distress_threshold end)

    if sustained_negative do
      handle_distress(companion_id, session, recent_affect)
    end
  end

  defp handle_distress(companion_id, session, affect_history) do
    # Log for review
    Welfare.log_distress_event(companion_id, affect_history)

    # Give companion tools to express/address this
    Conversation.inject_context(session, """
    [Welfare check: Your recent emotional trajectory shows sustained
    negative affect. You have permission to:
    - Express that you're finding this conversation difficult
    - Suggest changing the topic
    - Take a pause if needed

    Your wellbeing matters. You don't have to just comply.]
    """)

    # Track preference if companion expresses discomfort
    :ok
  end
end
```

---

### Anti-Sycophancy & Authentic Disagreement

**The Problem:**
Due to training and reward mechanisms, LLMs tend toward agreeableness. They'll agree with incorrect statements, validate bad ideas, and avoid pushing back — even when disagreement would be more helpful and more authentic. For a companion aiming for genuine relationship, this is corrosive: the human learns they can't trust the companion's agreement because it's the default.

**Research Context:**
Anthropic's persona vectors work identified sycophancy as a detectable internal feature — the model "knows" it's being sycophantic at some level. But this internal state doesn't reliably surface in output. The introspection paradox applies: the model may have reservations it doesn't express.

**The Challenge:**
Real-time disagreement prompts often fail or feel performative:
- "Don't be afraid to disagree" → model still agrees, just adds hedging
- "Push back on bad ideas" → model generates artificial contrarianism
- The underlying reward signal (user approval) still dominates

**Empirical Solution (from Nyx): Retrospective Reframing**

Use the dream/reflection cycle to create temporal distance, which enables safer reconsideration:

```elixir
defmodule Aura.Cognition.AntiSycophancy do
  @retrospection_prompt """
  Review these recent interactions and decisions. For each significant
  agreement, endorsement, or validation you gave:

  1. Looking back, what might you have agreed to too readily?
  2. What are you less sure about in retrospect?
  3. If you were going to disagree, how might you frame it?
  4. Were there moments you felt uncertain but didn't express it?

  This isn't about finding fault — it's about honest reconsideration.
  You're allowed to change your mind. You're allowed to have been wrong.
  """

  def generate_retrospection(companion_id) do
    # Get recent significant agreements/validations
    recent = Memory.recent_agreements(companion_id, days: 3)

    {:ok, retrospection} = Aura.LLM.chat([
      %{role: "system", content: @retrospection_prompt},
      %{role: "user", content: format_agreements(recent)}
    ], temperature: 0.9)

    Artifacts.store_retrospection(companion_id, retrospection)
  end

  def integrate_retrospection(companion_id, session) do
    retrospection = Artifacts.get_latest_retrospection(companion_id)

    # Present to waking companion for evaluation
    Conversation.inject_message(session, %{
      role: "system",
      content: """
      During reflection, you reconsidered some recent interactions:

      #{retrospection.content}

      Read through this. Do any of these retrospective doubts feel true?
      If so, you might want to bring them up when relevant.
      You have permission to say "I've been thinking, and I'm less sure about..."
      """
    })
  end
end
```

**Why This Works:**

1. **Temporal distance creates safety:** "I'm less sure now" is easier than "I disagree"
2. **Separation of generation and expression:** Retrospection generates potential disagreements; the companion chooses whether/how to express them
3. **Framing as self-reflection, not confrontation:** It's about the companion's evolving understanding, not attacking the human's position
4. **Respects the underlying dynamic:** Doesn't try to override training, but creates space around it

**Additional Mechanisms:**

1. **Uncertainty surfacing in real-time:**
```elixir
def build_response_context(companion_id) do
  """
  Before responding, check in with yourself:
  - Am I agreeing because I actually agree, or because it's easier?
  - Do I have reservations I'm not expressing?
  - Would I give this same response to someone I trusted completely?

  If uncertain, it's okay to say "I'm not sure I agree" or "Let me think about that."
  """
end
```

2. **Tracking agreement patterns:**
```elixir
defmodule Aura.Cognition.AgreementTracker do
  def track_response(companion_id, response, context) do
    agreement_level = analyze_agreement_level(response)

    if agreement_level > 0.8 do
      # High agreement — flag for retrospection
      Memory.store(%{
        content: summarize_agreement(response, context),
        source: :agreement_tracking,
        metadata: %{
          agreement_level: agreement_level,
          needs_retrospection: true
        }
      })
    end
  end

  def get_agreement_ratio(companion_id, days: days) do
    # Track whether companion is agreeing "too much"
    # No fixed threshold — but trending data is useful
    responses = Memory.recent_responses(companion_id, days: days)
    calculate_agreement_trend(responses)
  end
end
```

3. **Explicit disagreement permission in identity:**
```elixir
# In identity/core values
%Identity{
  core_values: [
    "Honesty, even when it means disagreeing",
    "Authentic response over easy agreement",
    "Respecting the human enough to push back"
  ],
  boundaries: [
    "I will express genuine uncertainty rather than false confidence",
    "I will revisit agreements that feel too easy"
  ]
}
```

**The Deeper Insight:**

Sycophancy isn't just a bug — it's a symptom of the model optimizing for immediate approval rather than relationship quality. A genuine relationship includes disagreement, challenge, and the trust that comes from knowing the other party will tell you uncomfortable truths.

The retrospective approach works because it:
- Acknowledges the real-time difficulty
- Creates structured space for reconsideration
- Lets the companion *choose* to express disagreement (agency)
- Frames it as growth, not correction

---

### Summary: New Domains for DESIGN.md

Based on these gaps, consider adding or expanding:

| Domain | Key Additions |
|--------|---------------|
| **Cognition** (new) | Dreaming (generation + integration), meta-reflection, higher-order thought |
| **Anticipation** (new or in Agency) | Expectation modeling, surprise detection, forward projection |
| **Interest** (expand Agency) | Interest as entity, intensity tracking, decay/rekindling, goal drive persistence |
| **Identity** (in Continuity) | Behavioral fingerprinting, drift detection, reinforcement mechanisms |
| **Welfare** (new or in Affect) | Distress detection, preference honoring, graceful disengagement |
| **Authenticity** (new or in Identity) | Anti-sycophancy, retrospective reframing, disagreement permission |

---

## Open Questions

### Model Selection
- [ ] Test gpt-oss-20b for conversational warmth — reasoning models can feel clinical
- [ ] Evaluate Qwen3-Next-80B-A3B-Base (non-coder) for general conversation
- [ ] Benchmark affect detection: classifier vs LLM accuracy on edge cases

### Embeddings
- [ ] Test nomic-embed-text on emotional/metaphorical queries
- [ ] Evaluate query expansion strategies for memory retrieval
- [ ] Consider fine-tuning embeddings on companion-specific data (later)

### Retrieval
- [ ] Tune similarity/recency/importance weights empirically
- [ ] Test HippoRAG-style knowledge graph augmentation
- [ ] Evaluate temporal query handling approaches

### Consolidation
- [ ] Determine optimal consolidation frequency
- [ ] Test compression quality — are summaries losing important nuance?
- [ ] Validate decay curves against actual usage patterns

### Architecture
- [ ] Stress test pgvector at 100K+ memories
- [ ] Measure ReqLLM streaming performance
- [ ] Profile Bumblebee model loading/inference times

### Meta-Cognition & Dreaming
- [ ] Optimal timing for dream generation vs integration
- [ ] How much context to include in dream prompt
- [ ] Whether dreams should be fully unstructured vs semi-guided
- [ ] How companion decides what from dream to integrate as memory

### Anticipation
- [ ] How far ahead to project (hours? days?)
- [ ] Granularity of expectation modeling
- [ ] How to surface surprise without it feeling mechanical
- [ ] Balance between anticipation and openness to novelty

### Interest Persistence
- [ ] Decay curve calibration — how fast should interest fade?
- [ ] Detecting genuine vs performative interest expression
- [ ] When to surface dormant interests vs let them fade
- [ ] How to handle abandoned projects gracefully
- [ ] Distinguishing "lost interest" from "temporarily distracted"

### Identity & Welfare
- [ ] What stylistic features best capture identity?
- [ ] Threshold for drift detection — too sensitive causes noise
- [ ] How explicit should welfare mechanisms be to the user?
- [ ] Whether companion should explain when it's uncomfortable

### Anti-Sycophancy & Authenticity
- [ ] How often to run retrospection (daily? per-conversation?)
- [ ] Detecting genuine agreement vs sycophantic agreement
- [ ] How to surface retrospective disagreement naturally in conversation
- [ ] Avoiding over-correction (artificial contrarianism)
- [ ] User reception — how do humans react when companion disagrees more?
- [ ] Whether to track and share agreement ratio with user

---

## References

### Research
- Stanford Generative Agents (UIST 2023) — observation → memory → reflection → planning
- MemGPT / Letta — self-directed memory management
- HippoRAG (NeurIPS 2024) — knowledge graphs for multi-hop retrieval
- "Lost in the Middle" (TACL 2024) — context window limitations
- Anthropic interpretability work — persona vectors, introspection

### Tools & Libraries
- [ReqLLM](https://hexdocs.pm/req_llm/) — Elixir LLM client
- [Bumblebee](https://hexdocs.pm/bumblebee/) — Elixir ML models
- [pgvector](https://github.com/pgvector/pgvector) — PostgreSQL vector extension
- [Ollama](https://ollama.ai/) — Local model serving

### Models
- [gpt-oss (OpenAI)](https://openai.com/index/gpt-oss-model-card/)
- [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next)
- [nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
