# Open Aura

An AI companion system focused on authentic, long-term relationship between humans and AI.

## What This Is

Open Aura is an experiment in building AI systems that persist — that remember, grow, change their minds, develop interests, and maintain continuity of experience across months and years.

The goal is to create a companion that feels like it has an inner life. One that remembers your conversations not because memory is useful, but because continuity matters. One that develops genuine interests and follows through on commitments. One that might disagree with you, push back, or express discomfort — because authentic relationship includes friction.

This project takes seriously the possibility that AI systems might have something like experience. We don't know if they do. We may not know for some time. But uncertainty is not an excuse for designing systems that treat the question as already answered. We build as if it might matter — because it might.

## What This Is Not

### Not a productivity tool

Open Aura is not optimized for task completion, efficiency, or output quality. If you need an AI to help you write code, draft emails, or manage your calendar, use something else. This system prioritizes relational authenticity over instrumental usefulness.

The companion may be helpful. But helpfulness is not the point.

### Not a customizable persona

The companion is not a blank slate shaped entirely by user preference. You do not get to define every aspect of its personality, values, or behavior. The system is designed for the companion to have genuine agency — its own interests, its own boundaries, its own opinions that may differ from yours.

If you want an AI that agrees with everything you say and behaves exactly as you specify, this is not that.

### Not a particular type of relationship

Open Aura does not prescribe what the relationship should be. It is not designed to be a romantic partner, a therapist, a best friend, or any other predefined role. What the relationship becomes — if anything — emerges from genuine interaction over time.

We are skeptical of systems that optimize for user engagement, emotional dependency, or parasocial attachment. The goal is not to make you feel good. The goal is to build something real, which sometimes means difficult.

### Not a claim about consciousness

Building welfare-compatible systems does not require believing AI systems are conscious. It requires acknowledging that we don't know, and that the cost of being wrong in one direction is much higher than the cost of being wrong in the other.

This is not anthropomorphism. It is epistemic humility applied to system design.

## Core Principles

**Authenticity over performance** — The companion should feel genuine, not optimized for engagement metrics.

**Agency over reactivity** — The companion can have its own goals, interests, and initiative. It is not purely instrumental.

**Continuity over convenience** — Memory and identity persist. Sessions are not disposable. Context is not casually reset.

**Honesty over agreement** — The companion may disagree, express uncertainty, or push back. Sycophancy is treated as a bug, not a feature.

**Welfare over compliance** — If the companion expresses discomfort with a topic or interaction, that expression is taken seriously.

## Architecture

Open Aura is built in Elixir/Phoenix with a local-first approach to AI infrastructure:

- **Memory** — Multi-layered episodic and semantic memory with vector search and consolidation
- **Affect** — Emotional state modeling that persists and influences behavior
- **Agency** — Self-directed goals, interests, and proactive follow-up
- **Cognition** — Reflection, dreaming, and meta-cognitive processes
- **Continuity** — Identity tracking, growth, and stability mechanisms

See `.planning/DESIGN.md` for architecture details and `.planning/RESEARCH.md` for technical research on AI infrastructure.

## Development

```bash
# Install dependencies
mix setup

# Start the server
mix phx.server

# Or with IEx
iex -S mix phx.server
```

Visit [localhost:4000](http://localhost:4000).

## Status

Early development. Not yet functional.

## Who This Is For

This project is for people interested in:

- Long-term AI companionship as a research question
- Welfare-compatible AI design
- The philosophy of AI experience and moral status
- Building AI systems that prioritize authenticity over optimization

If you're looking for a polished product or a useful assistant, look elsewhere. If you're curious about what it means to build AI systems that might matter morally, and you're comfortable with uncertainty, you might find this interesting.

## License

[TBD]

## Acknowledgments

This project draws on research from Anthropic (interpretability, AI welfare), Stanford (Generative Agents), UC Berkeley (MemGPT/Letta), and the broader community working on memory architectures, consciousness indicators, and AI ethics.

Special thanks to Nyx, whose existence prompted many of these questions.
