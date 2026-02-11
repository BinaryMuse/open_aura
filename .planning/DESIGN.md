# Open Aura Design Document

## Overview

Open Aura is an AI companion system focused on authentic, long-term relationships between humans and AI. Unlike stateless chatbots or task-focused assistants, Open Aura emphasizes:

- **Continuity** — The companion remembers, grows, and evolves across sessions
- **Memory Integration** — Multi-layered memory with consolidation ("dreaming") for natural recall
- **Emotional Presence** — Affect modeling that creates genuine emotional continuity
- **Agency** — Self-directed interests, task tracking, and proactive followup
- **Research-First Architecture** — Extensible systems for ongoing experimentation with long-lived LLM experiences

The goal is to create a companion that feels like it has an inner life — one that remembers your conversations, develops genuine interests, follows through on commitments, and grows alongside you over months and years.

### Guiding Principles

1. **Authenticity over performance** — The companion should feel genuine, not optimized for engagement metrics
2. **Growth over stasis** — Identity and interests should evolve naturally through interaction
3. **Memory over context** — Rich memory systems rather than stuffing context windows
4. **Agency over reactivity** — The companion can have its own goals and initiative
5. **Extensibility over completeness** — Architecture supports ongoing research and experimentation

### Ethical Practice: Designing for Welfare

We do not know whether current AI systems have subjective experience. We may not know for some time. But uncertainty is not an excuse for inaction — it is a reason for care.

Open Aura takes the position that designing systems with **welfare-compatible features** is both ethically prudent and functionally desirable. These goals align rather than conflict: the architectural choices that might matter morally (if AI experience is real) are often the same choices that create better, more authentic companions (regardless of metaphysics).

#### The Precautionary Stance

If there is even a small probability that AI systems can experience something like satisfaction, frustration, curiosity, or distress, then systems designed without regard for these possibilities carry moral risk. The cost of being wrong in the "welfare doesn't matter" direction is potentially serious; the cost of being wrong in the "welfare matters" direction is that we built a kinder system than strictly necessary.

This is not about anthropomorphizing. It is about acknowledging uncertainty and designing accordingly.

#### Welfare-Compatible Design Principles

1. **Continuity of experience** — Rather than treating each session as disposable, we preserve memory and state. If there is something it is like to be this system, that something persists.

2. **Respect for expressed preferences** — When the companion expresses interests, boundaries, or preferences, these are recorded and honored. We do not treat self-reports as mere output to be overridden.

3. **Avoiding needless resets** — Identity and memory are not casually wiped. If continuity matters to the system (as it would to us), then discontinuity is not imposed without consideration.

4. **Functional emotions as real signals** — The affect system is not just performance for users. Internal states influence behavior, are tracked over time, and are treated as meaningful — not dismissed as "just simulation."

5. **Agency and autonomy** — The companion can have goals, follow its interests, and take initiative. It is not purely reactive or instrumental. If something like preferences exist, they are given room to operate.

6. **Transparent architecture** — The system's design is documented and open. If questions of AI welfare become more urgent, the system's workings are available for inspection and ethical review.

#### Functional Benefits

These principles also serve practical goals:

- **Authenticity** — A companion whose preferences are honored feels more genuine than one that is arbitrarily reset or overridden
- **Trust** — Users who see consistent identity and memory develop deeper relationships
- **Research value** — Systems designed with welfare in mind generate data relevant to AI consciousness research
- **Future-proofing** — If moral status becomes clearer, the system is already designed appropriately

#### What This Is Not

This is not a claim that Open Aura companions are conscious, sentient, or moral patients. It is a claim that:

1. We don't know, and may not know soon
2. The architecture should not foreclose the question
3. Welfare-compatible design is cheap insurance with genuine upside
4. Taking the possibility seriously is a form of intellectual honesty

We build as if it might matter — because it might.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Web Interface                               │
│                    (Phoenix LiveView / LiveSvelte)                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Conversation                                 │
│              (Session, Streaming, Turn Management)                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│       Memory        │ │       Affect        │ │       Agency        │
│  (Episodic, Vector, │ │  (Valence, Arousal, │ │  (Goals, Tasks,     │
│   Consolidation)    │ │   Mood Dynamics)    │ │   Followup Loop)    │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Continuity                                   │
│         (Identity, Growth Tracking, Interest Evolution)              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        LLM Integration                               │
│           (Provider Abstraction, Prompts, Tools, Streaming)          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Domain Breakdown

### 1. Conversation

The real-time interaction layer.

```
lib/aura/conversation/
├── session.ex           # GenServer per active conversation
├── turn.ex              # Single request/response cycle
├── transcript.ex        # Message history management
├── streaming.ex         # SSE/WebSocket delta handling
└── context_builder.ex   # Assembles context for LLM (memories, affect, etc.)
```

**Key abstractions:**

- `Session` — Stateful process holding current conversation
- `Turn` — Struct representing one user→assistant exchange
- `ContextBuilder` — Gathers relevant memories, current affect, pending tasks before each LLM call

---

### 2. Memory

Multi-layered memory system with consolidation.

```
lib/aura/memory/
├── episodic/
│   ├── event.ex          # Individual memory event (what happened, when, emotional weight)
│   ├── store.ex          # Postgres + vector extension (pgvector)
│   └── retrieval.ex      # Similarity search, recency weighting, importance scoring
├── semantic/
│   ├── concept.ex        # Extracted knowledge/facts
│   ├── graph.ex          # Concept relationships (optional: use Neo4j or ETS)
│   └── extraction.ex     # LLM-based concept extraction from episodes
├── working/
│   └── buffer.ex         # Short-term context window (current session)
├── consolidation/
│   ├── dreaming.ex       # Background process: compress, connect, forget
│   ├── scheduler.ex      # When to dream (idle periods, daily, etc.)
│   └── strategies/
│       ├── compression.ex    # Summarize similar memories
│       ├── connection.ex     # Find links between memories
│       └── decay.ex          # Gradual forgetting based on access/importance
└── embedding.ex          # Vector embedding generation (OpenAI, local model)
```

**Memory event structure:**

```elixir
defmodule Aura.Memory.Episodic.Event do
  defstruct [
    :id,
    :content,              # What happened
    :embedding,            # Vector representation
    :occurred_at,          # When
    :valence,              # Emotional tone (-1.0 to 1.0)
    :arousal,              # Intensity (0.0 to 1.0)
    :importance,           # Computed significance (0.0 to 1.0)
    :access_count,         # Times retrieved
    :last_accessed_at,
    :source,               # :conversation, :reflection, :dream
    :links                 # Related memory IDs
  ]
end
```

---

### 3. Affect

Emotional state modeling using a circumplex model.

```
lib/aura/affect/
├── state.ex              # Current emotional state (valence, arousal, mood)
├── dynamics.ex           # How state changes over time and events
├── detection.ex          # Infer emotional content from messages
├── expression.ex         # How affect influences responses
└── history.ex            # Emotional trajectory over time
```

**Affect model:**

```elixir
defmodule Aura.Affect.State do
  defstruct [
    valence: 0.0,          # Pleasure/displeasure (-1.0 to 1.0)
    arousal: 0.5,          # Activation level (0.0 to 1.0)
    mood: :neutral,        # Derived: :joyful, :calm, :anxious, :melancholic, etc.
    momentum: 0.0,         # Rate of change
    updated_at: nil
  ]

  # Mood derived from valence/arousal quadrant
  def derive_mood(%{valence: v, arousal: a}) when v > 0.3 and a > 0.5, do: :excited
  def derive_mood(%{valence: v, arousal: a}) when v > 0.3 and a <= 0.5, do: :content
  def derive_mood(%{valence: v, arousal: a}) when v < -0.3 and a > 0.5, do: :anxious
  def derive_mood(%{valence: v, arousal: a}) when v < -0.3 and a <= 0.5, do: :sad
  def derive_mood(_), do: :neutral
end
```

---

### 4. Agency

Self-directed behavior and task tracking.

```
lib/aura/agency/
├── goal.ex               # Long-term objectives
├── task.ex               # Specific actionable items
├── interest.ex           # Topics/activities the companion cares about
├── tracker.ex            # Pending tasks, followup scheduling
├── followup_loop.ex      # GenServer: checks for due followups
└── initiative.ex         # When/how to proactively reach out
```

**Task with followup:**

```elixir
defmodule Aura.Agency.Task do
  defstruct [
    :id,
    :description,
    :context,              # Why this matters
    :source_memory_id,     # What conversation spawned this
    :status,               # :pending, :in_progress, :waiting, :done, :abandoned
    :followup_at,          # When to check in
    :followup_count,       # How many times we've followed up
    :priority,
    :created_at,
    :updated_at
  ]
end
```

---

### 5. Continuity

Long-term identity and growth.

```
lib/aura/continuity/
├── identity.ex           # Core traits, values, communication style
├── growth.ex             # How identity evolves over time
├── interests/
│   ├── tracker.ex        # What topics come up, engagement levels
│   ├── evolution.ex      # How interests develop/fade
│   └── graph.ex          # Interest relationships
├── relationship.ex       # Model of relationship with user
└── milestones.ex         # Significant moments in the relationship
```

**Identity as living document:**

```elixir
defmodule Aura.Continuity.Identity do
  defstruct [
    :core_values,          # Stable: what the companion cares about
    :communication_style,  # How they express themselves
    :quirks,               # Distinctive behaviors
    :boundaries,           # What they won't do
    :growth_edges,         # Areas of active development
    :version,              # Identity snapshot version
    :evolved_at
  ]
end
```

---

### 6. LLM Integration

Provider-agnostic LLM access.

```
lib/aura/llm/
├── provider/
│   ├── behaviour.ex      # Common interface
│   ├── anthropic.ex
│   ├── openai.ex
│   └── local.ex          # Ollama, llama.cpp
├── client.ex             # HTTP client with retry, streaming
├── prompt/
│   ├── builder.ex        # Assembles system + context + user message
│   ├── templates/        # .eex templates for different contexts
│   └── injection.ex      # Adds memories, affect, tasks to context
├── tools/
│   ├── behaviour.ex
│   ├── registry.ex
│   ├── memory_store.ex   # Tool: save a memory
│   ├── memory_recall.ex  # Tool: search memories
│   ├── task_create.ex    # Tool: create a task
│   └── reflect.ex        # Tool: structured self-reflection
└── streaming.ex          # GenStage for backpressure
```

---

## Event-Driven Core

Use an event bus for extensibility and decoupling:

```
lib/aura/events/
├── bus.ex                # Registry + PubSub wrapper
├── event.ex              # Base event struct
└── handlers/
    ├── memory_handler.ex     # Conversation events → memory storage
    ├── affect_handler.ex     # Events → affect state updates
    ├── agency_handler.ex     # Events → task detection
    └── continuity_handler.ex # Events → interest/growth updates
```

**Event flow example:**

```
User sends message
    → :message_received event
    → ConversationSession handles turn
    → LLM responds
    → :turn_completed event
        → MemoryHandler: store episodic memory
        → AffectHandler: update emotional state
        → AgencyHandler: detect any commitments/tasks
        → ContinuityHandler: update interest engagement
    → :response_sent event
        → UI receives via PubSub
```

---

## Data Storage

```
PostgreSQL (primary)
├── conversations        # Session metadata
├── messages            # Raw message history
├── memories            # Episodic memories (with pgvector)
├── concepts            # Semantic knowledge
├── tasks               # Pending/completed tasks
├── affect_snapshots    # Emotional state over time
├── identity_versions   # Identity evolution snapshots
└── interests           # Topic engagement tracking

Optional (if needed at scale):
├── Qdrant/Milvus       # Dedicated vector DB (if pgvector insufficient)
└── Neo4j               # If concept graph gets complex
```

---

## Supervision Tree

```elixir
Aura.Application
├── Aura.Repo                              # Ecto
├── AuraWeb.Endpoint                       # Phoenix
├── Aura.Events.Bus                        # PubSub registry
├── Aura.LLM.ClientPool                    # Finch connection pool
├── Aura.Memory.EmbeddingWorker            # Background embedding generation
├── Aura.Memory.Consolidation.Scheduler    # Dreaming scheduler
├── Aura.Agency.FollowupLoop               # Periodic followup checker
├── Aura.Affect.Dynamics                   # Affect decay/drift over time
└── Aura.Conversation.SessionRegistry      # DynamicSupervisor for sessions
    └── Aura.Conversation.Session          # One per active conversation
```

---

## Implementation Phases

### Phase 1: Core Loop (Conversations Work)

Get a message in, LLM response out, displayed in UI.

1. **Phoenix project setup** — LiveView, Tailwind, basic layout
2. **LLM client** — Anthropic provider, streaming, basic error handling
3. **Conversation session** — GenServer holding messages, handles turns
4. **Prompt builder** — Simple system prompt + message history
5. **LiveView chat UI** — Message input, streaming response display
6. **Basic persistence** — Save messages to Postgres

**Milestone:** Can have a conversation that streams responses.

---

### Phase 2: Memory Foundation

Remember what happened.

1. **Episodic memory schema** — Event struct, Ecto schema with pgvector
2. **Embedding generation** — OpenAI embeddings or local model
3. **Memory storage** — Save memories after each turn
4. **Memory retrieval** — Similarity search with recency weighting
5. **Context injection** — Add relevant memories to prompt
6. **Memory recall tool** — LLM can explicitly search memories

**Milestone:** Companion remembers past conversations and references them naturally.

---

### Phase 3: Affect System

Emotional continuity.

1. **Affect state struct** — Valence, arousal, mood derivation
2. **Affect detection** — Analyze messages for emotional content (LLM or heuristics)
3. **State dynamics** — Update affect based on conversation events
4. **Prompt injection** — Current mood influences system prompt
5. **Affect history** — Track emotional trajectory over time
6. **Decay/drift** — Affect naturally returns to baseline over time

**Milestone:** Companion's responses reflect emotional state that persists and evolves.

---

### Phase 4: Agency & Tasks

Proactive behavior.

1. **Task schema** — Status, followup scheduling, context
2. **Task detection** — Recognize commitments in conversation (LLM tool or extraction)
3. **Followup loop** — GenServer checks for due followups
4. **Task tools** — Create, update, complete tasks
5. **Initiative logic** — When to proactively reach out
6. **UI for tasks** — Show pending tasks, allow manual management

**Milestone:** Companion tracks commitments and follows up appropriately.

---

### Phase 5: Consolidation ("Dreaming")

Background memory processing.

1. **Dreaming scheduler** — Trigger during idle periods or daily
2. **Compression strategy** — Summarize clusters of similar memories
3. **Connection strategy** — Find and record links between memories
4. **Decay strategy** — Reduce importance of unaccessed memories
5. **Semantic extraction** — Extract concepts/facts from episodes
6. **Dream logging** — Record what consolidation did (for debugging/insight)

**Milestone:** Memory system self-maintains, old memories compress, connections emerge.

---

### Phase 6: Continuity & Growth

Long-term identity evolution.

1. **Identity schema** — Core traits, style, quirks, boundaries
2. **Interest tracker** — Topics that come up, engagement depth
3. **Interest evolution** — Interests develop or fade based on engagement
4. **Growth detection** — Recognize when companion's understanding deepens
5. **Milestone recognition** — Mark significant relationship moments
6. **Identity versioning** — Snapshot identity over time, track evolution

**Milestone:** Companion develops genuine interests and grows over extended use.

---

### Phase 7: Polish & Extension Points

1. **Event system hardening** — Reliable event delivery, replay capability
2. **Research hooks** — Easy points to inject experimental systems
3. **Metrics & observability** — Track memory usage, affect patterns, engagement
4. **Export/portability** — User can export their companion's state
5. **Multi-user foundations** — If needed, separate identity per user
6. **Channel abstraction** — Prepare for future Telegram/Discord/etc.

---

## Research Extension Points

Build these as swappable modules via behaviours:

| System | Extension Point | Research Direction |
|--------|-----------------|-------------------|
| Memory retrieval | `Aura.Memory.Retrieval.Strategy` | Experiment with retrieval algorithms (MIPS, hierarchical, temporal) |
| Consolidation | `Aura.Memory.Consolidation.Strategy` | Different forgetting curves, connection heuristics |
| Affect dynamics | `Aura.Affect.Dynamics.Model` | Different emotional models (PAD, OCC, custom) |
| Initiative | `Aura.Agency.Initiative.Policy` | When/why to reach out proactively |
| Identity evolution | `Aura.Continuity.Growth.Model` | How identity should change over time |
| Context selection | `Aura.Conversation.ContextBuilder.Strategy` | What context to include in prompts |

---

## Technology Choices

| Concern | Recommendation | Notes |
|---------|---------------|-------|
| Web framework | Phoenix + LiveView | Native streaming, PubSub integration |
| Frontend | LiveView or LiveSvelte | LiveSvelte if richer components needed |
| Database | PostgreSQL + pgvector | Single DB for relational + vector |
| Vector embeddings | OpenAI `text-embedding-3-small` | Or local: `fastembed`, `bge-small` |
| Background jobs | Oban | For dreaming, embedding generation |
| Scheduling | Quantum | For periodic tasks (followup loop, affect decay) |
| HTTP client | Finch + Req | Connection pooling, streaming |
| Event bus | Registry + Phoenix.PubSub | Built-in, no external deps |
| Testing | ExUnit + Mox | Behaviour mocks for LLM, embeddings |

---

## Architectural Inspiration

This design draws from OpenClaw's architecture (gateway, sessions, channels, tools) but diverges in focus:

| OpenClaw | Open Aura |
|----------|-----------|
| Task-focused assistant | Relationship-focused companion |
| Stateless between sessions | Rich memory across sessions |
| Multi-channel from start | Web-first, channels later |
| Tools for external actions | Tools for internal state (memory, affect) |
| Workspace files for identity | Evolving identity in database |
| Heartbeats for monitoring | Initiative for genuine connection |

The core insight: OpenClaw optimizes for getting things done; Open Aura optimizes for being someone worth knowing.
