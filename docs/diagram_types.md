# Diagram Types

Taxonomy of spatial diagram templates used in `docs/kanban/`. Each type uses a
generic set of spatial relations to depict how information is related.

Types 1-16: extracted from existing diagrams (66 instances across 15 files).
Types 17-28: extended catalog — spatial relationships not covered above.

---

## 1. Vertical Sequential Flow (16 instances)

Top-to-bottom chain of steps connected by arrows. The most common pattern.

```
  step A
    │
    ▼
  step B
    │
    ▼
  step C ──→ side effect
    │
    ▼
  step D
```

Used in: 01 (request/stream paths), 02 (entry, adapter, session mgr, state
transitions), 06 (event pipeline), 07 (launch config), 08 (config→bundle,
MCP OAuth), 10 (hydration), 12 (data flow), 13 (review auto-actions),
15 (onboarding, save orchestration, cline settings)

---

## 2. Record / Schema Box (8 instances)

Single box enumerating the fields or capabilities of a data type or module.

```
  ┌─ TypeName ──────────────────────────────────────┐
  │  fieldA : string            description          │
  │  fieldB : number            description          │
  │  fieldC : SubType | null    description          │
  │  nested : Map<K, V>         description          │
  └──────────────────────────────────────────────────┘
```

Used in: 03 (SessionEntry), 06 (ClineTaskSessionEntry, MessageRepository),
07 (ProviderService, SdkProviderBoundary, ResolvedClineLaunchConfig),
08 (RuntimeClineMcpServer), 10 (snapshot shape)

---

## 3. Layered Stack (5 instances)

Horizontally-spanning boxes stacked vertically, representing architectural
tiers. Arrows between layers show communication direction.

```
  ┌─ LAYER 1 ──────────────────────────────┐
  │  top-level concern                      │
  └──────────────────┬─────────────────────┘
                     │
                     ▼
  ┌─ LAYER 2 ──────────────────────────────┐
  │  middle concern                         │
  └──────────────────┬─────────────────────┘
                     │
                     ▼
  ┌─ LAYER 3 ──────────────────────────────┐
  │  low-level concern                      │
  └─────────────────────────────────────────┘
```

Used in: 01 (three-tier), 05 (session stack, provider stack),
14 (side-by-side stacks, shared four-layer pattern)

---

## 4. Definition List / Glossary (5 instances)

Independent stacked boxes, each a labeled term with a short description.
No arrows between them — items are peers.

```
  ┌─ term A ────────────────────────────────┐
  │  Short description of term A.           │
  └─────────────────────────────────────────┘
  ┌─ term B ────────────────────────────────┐
  │  Short description of term B.           │
  └─────────────────────────────────────────┘
  ┌─ term C ────────────────────────────────┐
  │  Short description of term C.           │
  └─────────────────────────────────────────┘
```

Used in: 01 (core concepts, ownership), 05 (supporting modules),
12 (hook groups), 13 (hook responsibilities)

---

## 5. Table / Matrix (5 instances)

Rows and columns comparing multiple items across shared dimensions.

```
  ┌──────────┬──────────┬──────────┬──────────┐
  │ Item     │ Dim A    │ Dim B    │ Dim C    │
  ├──────────┼──────────┼──────────┼──────────┤
  │ alpha    │ value    │ value    │ value    │
  ├──────────┼──────────┼──────────┼──────────┤
  │ beta     │ value    │ value    │ value    │
  ├──────────┼──────────┼──────────┼──────────┤
  │ gamma    │ value    │ value    │ value    │
  └──────────┴──────────┴──────────┴──────────┘
```

Used in: 04 (adapter matrix), 06 (event→effect dispatch),
09 (broadcast types), 10 (delta types, what-lives-where)

---

## 6. Sequence Diagram (4 instances)

Multiple actor columns with time flowing downward. Arrows show messages
or calls between actors.

```
  Actor A              Actor B              Actor C
  ───────              ───────              ───────
     │                    │                    │
     │  request           │                    │
     │───────────────────▶│                    │
     │                    │  delegate          │
     │                    │───────────────────▶│
     │                    │                    │
     │                    │  result            │
     │                    │◀───────────────────│
     │                    │                    │
     │  response          │                    │
     │◀───────────────────│                    │
```

Used in: 07 (OAuth flow), 09 (WebSocket protocol),
10 (persistence round-trip, conflict detection)

---

## 7. Fork / Decision Point (4 instances)

Binary YES/NO branch at a condition, splitting into two paths.
Often embedded within a vertical flow.

```
  input
    │
    ├─ condition? ──YES──→ path A
    │                      result A
    │
    └─ NO:
         path B
         result B
```

Used in: 01 (request path: cline?), 02 (entry: agent===cline?),
11 (runtime-API fork), 14 (home sidebar agent detection)

---

## 8. State Machine (3 instances)

Named states connected by labeled transition edges. Shows which events
cause which state changes.

```
             event X
    ┌──────────────────────┐
    │                      │
    v                      │
  ┌───────┐   event Y   ┌─┴─────┐
  │ stateA│ ───────────▶│ stateB │
  └───┬───┘             └────────┘
      │
      │ event Z
      ▼
  ┌────────┐
  │ stateC │
  └────────┘
```

Used in: 02 (session state machine), 03 (PTY state machine),
13 (card column transitions)

---

## 9. Tree / Hierarchy (3 instances)

Parent-to-children branching structure showing ownership or delegation.

```
  root
    │
    ├── child A ──→ leaf 1
    │               leaf 2
    │
    ├── child B ──→ leaf 3
    │
    └── child C    (lazy-loaded)
```

Used in: 09 (workspace registry, state hub event sources),
11 (router→API modules)

---

## 10. Side-by-Side Comparison (3 instances)

Two parallel structures shown adjacently to highlight symmetry or contrast.

```
  VARIANT A                         VARIANT B
  ─────────                         ─────────
  ┌────────────────────┐            ┌────────────────────┐
  │  transport layer   │            │  transport layer   │
  └────────┬───────────┘            └────────┬───────────┘
           │                                  │
           ▼                                  ▼
  ┌────────────────────┐            ┌────────────────────┐
  │  session layer     │            │  session layer     │
  └────────┬───────────┘            └────────┬───────────┘
           │                                  │
           ▼                                  ▼
  ┌────────────────────┐            ┌────────────────────┐
  │  component         │            │  component         │
  └────────────────────┘            └────────────────────┘
```

Used in: 07 (Kanban config vs SDK store), 14 (chat vs terminal stacks),
15 (config persistence split)

---

## 11. Nested UI Layout (2 instances)

Nested boxes representing visual component containment. Shows how UI
regions are composed spatially.

```
  ┌─ Shell ──────────────────────────────────────┐
  │  ┌─ Sidebar ──────┐  ┌─ Main ─────────────┐ │
  │  │  nav items      │  │  ┌─ Toolbar ─────┐ │ │
  │  │  agent panel    │  │  │  actions       │ │ │
  │  │                 │  │  └────────────────┘ │ │
  │  │                 │  │  ┌─ Content ──────┐ │ │
  │  │                 │  │  │  board / view  │ │ │
  │  │                 │  │  └────────────────┘ │ │
  │  └─────────────────┘  └────────────────────┘ │
  └──────────────────────────────────────────────┘
```

Used in: 12 (surface map), 15 (settings dialog sections)

---

## 12. Namespace / API Map (2 instances)

Nested boxes listing API endpoints or procedures grouped by namespace.

```
  appRouter
  ┌────────────────────────────────────────────┐
  │  ┌─ users ──────────────┐  ┌─ posts ────┐ │
  │  │  list                │  │  list       │ │
  │  │  getById             │  │  create     │ │
  │  │  create              │  │  update     │ │
  │  │  delete              │  │  delete     │ │
  │  └──────────────────────┘  └────────────┘  │
  └────────────────────────────────────────────┘
```

Used in: 11 (namespace diagram, capability summary)

---

## 13. Lifecycle Phases (2 instances)

Horizontal chain of named phase boxes showing a progression.

```
  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
  │ phase 1 │───▶│ phase 2 │───▶│ phase 3 │───▶│ phase 4 │
  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
    details        details        details        details
```

Used in: 05 (START→SEND→STOP→RESUME),
09 (Connect→Snapshot→Deltas→Disconnect)

---

## 14. Boundary / Containment Region (2 instances)

Two regions separated by a hard boundary marker, emphasizing that
code or data must not cross the line directly.

```
  ┌─────────────────────────────────────────────┐
  │  APPLICATION CODE                            │
  │  moduleA, moduleB, moduleC                   │
  │  These NEVER import sdk/* directly.          │
  │                                              │
  ╞══════════════════════════════════════════════╡
  │  ▓▓▓▓▓▓▓▓▓▓  HARD BOUNDARY  ▓▓▓▓▓▓▓▓▓▓▓▓  │
  ╞══════════════════════════════════════════════╡
  │                                              │
  │  sdk-boundary.ts  ← sole import site         │
  │  Re-exports all SDK types as aliases         │
  └─────────────────────────────────────────────┘
```

Used in: 05 (app code vs SDK), 07 (secrets boundary)

---

## 15. Horizontal Processing Pipeline (1 instance)

Left-to-right chain of processing stages, each transforming data
and passing it to the next.

```
  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
  │  source  │────▶│ filter A │────▶│ filter B │────▶│  sink    │
  └──────────┘     └──────────┘     └──────────┘     └──────────┘
```

Used in: 02 (PTY→Protocol Filter→State Mirror→WS Bridge→Browser)

---

## 16. Timeline (1 instance)

Events plotted at specific time points along a horizontal axis.

```
  ──────────────────────────────────────────────────────────
  t=0ms    event arrives
           │  start timer
           │
  t=50ms   second event arrives
           │  timer already running
           │
  t=150ms  timer fires → flush
  ──────────────────────────────────────────────────────────
```

Used in: 09 (batching timeline: 150ms batch window)

---

# Extended Catalog

Diagram types not yet used in the codebase, each filling a specific gap
in what the 16 types above can express.

---

## 17. Fan-Out / Fan-In

Models one-to-many dispatch (fan-out) or many-to-one aggregation (fan-in).
Existing flows are linear chains; this captures the spatial shape of
broadcast, scatter-gather, or Promise.all patterns where **parallelism
is the point**, not just a fork that never reconverges.

```
                          ┌──────────┐
                     ┌───▶│ worker A │───┐
                     │    └──────────┘   │
  ┌──────────┐       │    ┌──────────┐   │    ┌───────────┐
  │ dispatch │───────┼───▶│ worker B │───┼───▶│ aggregate │
  └──────────┘       │    └──────────┘   │    └───────────┘
                     │    ┌──────────┐   │
                     └───▶│ worker C │───┘
                          └──────────┘
```

Niche: event emitter fan-out, parallel Promise.all, MapReduce,
WebSocket broadcast to N clients.

---

## 18. Ring / Cycle

A closed loop where the output of the last step feeds back into the
first. Sequential Flow implies a start and end; this captures
**steady-state loops** like event loops, poll-process-emit cycles,
or retry-until-done patterns.

```
       ┌──────────┐
  ┌───▶│  check   │───┐
  │    └──────────┘   │
  │                    ▼
  │    ┌──────────┐  ┌──────────┐
  │    │  apply   │◀─│  decide  │
  │    └────┬─────┘  └──────────┘
  │         │
  └─────────┘
```

Niche: event loops, reconciliation cycles, retry loops,
poll-driven sync, watch-rebuild-reload dev servers.

---

## 19. Swimlane

Like a Sequence Diagram but with **horizontal lanes** rather than
vertical actor columns. Better when the emphasis is on which
**layer or process** owns each step, rather than message ordering.
Time flows left-to-right; ownership is top-to-bottom.

```
  ┌─ Browser ──────┬───────────────┬────────────────┬──────────┐
  │  click          │               │                │  render  │
  └────────┬────────┴───────────────┴────────────────┴──────────┘
           │                                         ▲
  ┌────────▼────────┬───────────────┬────────────────┴──────────┐
  │  TRPC           │  validate     │  return result            │
  └─ Server ────────┴───────┬───────┴───────────────────────────┘
                            │               ▲
  ┌─────────────────────────▼───────┬───────┴───────────────────┐
  │  DB                     query   │  rows                     │
  └─────────────────────────────────┴───────────────────────────┘
```

Niche: cross-cutting request traces where you want to see both
who owns each step and the temporal order.

---

## 20. Dependency Graph (DAG)

A directed acyclic graph where edges mean "depends on" or "imports."
Trees enforce single-parent; this allows **multiple parents**,
capturing the real shape of module import graphs, build dependency
order, or task prerequisite chains.

```
          ┌───┐
     ┌───▶│ A │◀───┐
     │    └───┘    │
  ┌──┴──┐       ┌──┴──┐
  │  B  │       │  C  │
  └──┬──┘       └──┬──┘
     │    ┌───┐    │
     └───▶│ D │◀───┘
          └───┘
```

Niche: import graphs, task dependency chains (kanban linked tasks),
build order, migration ordering.

---

## 21. Venn / Overlap Region

Shows **shared membership** between two or more sets. No existing
type captures the concept of intersection — what belongs to both
A and B but neither alone.

```
     ┌─────────────────────┐
     │  Module A            │
     │          ┌───────────┼──────────────┐
     │          │ shared    │  Module B    │
     │          │ types +   │              │
     │          │ utilities │              │
     └──────────┼───────────┘              │
                │                          │
                └──────────────────────────┘
```

Niche: shared type re-exports between boundaries, feature flag
overlap, permission intersection, API surface shared across versions.

---

## 22. Adapter / Port-and-Plug

A central interface (port) with multiple interchangeable
implementations plugged in. The Table/Matrix can list adapters,
but this captures the **spatial contract**: one stable socket,
N swappable plugs.

```
                      ┌──────────────────┐
  ┌───────────┐       │                  │
  │  Impl A   │──────▶│                  │
  └───────────┘       │    Interface     │
  ┌───────────┐       │    (port)        │──────▶ consumer
  │  Impl B   │──────▶│                  │
  └───────────┘       │                  │
  ┌───────────┐       │                  │
  │  Impl C   │──────▶│                  │
  └───────────┘       └──────────────────┘
```

Niche: strategy pattern, driver/adapter registries, provider
catalogs, plugin systems.

---

## 23. Onion / Concentric Rings

Nested rings where each layer wraps the one inside it. Unlike a
Layered Stack (which implies top-to-bottom flow), this emphasizes
**encapsulation depth**: the innermost ring is the most protected,
and access must traverse every surrounding layer.

```
  ┌─────────────────────────────────────────────┐
  │  middleware / interceptors                   │
  │  ┌─────────────────────────────────────┐    │
  │  │  application logic                   │    │
  │  │  ┌─────────────────────────────┐    │    │
  │  │  │  domain core / pure model   │    │    │
  │  │  └─────────────────────────────┘    │    │
  │  └─────────────────────────────────────┘    │
  └─────────────────────────────────────────────┘
```

Niche: middleware stacks (Koa, Express), hexagonal / clean
architecture, security zones, error-handling wrappers.

---

## 24. Before / After Diff

Two snapshots placed side-by-side or top-to-bottom with markers
showing what changed. No existing type captures **delta between
two states of the same structure**.

```
  BEFORE                            AFTER
  ┌──────────────────────┐          ┌──────────────────────┐
  │  config.json         │          │  config.json         │
  │                      │          │                      │
  │  agent: "claude"     │          │  agent: "claude"     │
  │  mode:  "plan"       │    ──▶   │  mode:  "auto"     ◀── changed
  │                      │          │  retry: 3           ◀── added
  └──────────────────────┘          └──────────────────────┘
```

Niche: migration guides, config changes, schema evolution,
refactoring plans, state transitions with payload diffs.

---

## 25. Priority / Weighted Ranking

An ordered vertical list where position encodes **rank or priority**,
and optional bar-width or annotation encodes magnitude. Definition
Lists show peers; this shows that order matters.

```
  ▐████████████████████████████████▌  1. startTaskSession   (42 calls)
  ▐████████████████████████▌          2. stopTaskSession    (31 calls)
  ▐█████████████████▌                 3. sendInput          (22 calls)
  ▐████████▌                          4. getConfig          (11 calls)
  ▐███▌                               5. resetState         ( 4 calls)
```

Niche: hot-path ranking, error frequency, API call frequency,
performance bottleneck prioritization.

---

## 26. Bidirectional / Duplex Channel

Two entities connected by **two contra-flowing arrows**, showing
that communication goes both ways with different semantics in
each direction. A single arrow between boxes loses this.

```
  ┌──────────────┐                      ┌──────────────┐
  │              │  commands / writes    │              │
  │    Client    │─────────────────────▶│    Server    │
  │              │                      │              │
  │              │◀─────────────────────│              │
  │              │  events / streams    │              │
  └──────────────┘                      └──────────────┘
```

Niche: WebSocket IO+control channels, request/response with
push notifications, TRPC mutations vs subscriptions,
stdin/stdout pairs.

---

## 27. Escape Hatch / Exception Path

A main flow with a **diagonal breakout** showing the error or
fallback path that departs from the happy path. Fork/Decision
is a planned branch; this emphasizes that the departure is
exceptional and usually undesirable.

```
  step A
    │
    ▼
  step B
    │
    ├──────╳ error ──────▶ fallback path
    │                        │
    ▼                        ▼
  step C                  log + recover
    │                        │
    ▼                        │
  step D ◀───────────────────┘
```

Niche: try/catch flows, circuit breakers, retry-with-fallback,
graceful degradation, timeout recovery.

---

## 28. Resource Lifecycle (Acquire-Use-Release)

Three-phase pattern showing a resource being **acquired, used
under a scope, then released** — with the release guaranteed
even on error. This is structurally different from a plain
flow because the release is spatially tied back to the acquire.

```
  ┌─ acquire ─────────────────────────────────────┐
  │  const handle = open(resource)                 │
  │                                                │
  │  ┌─ use (scoped) ──────────────────────────┐  │
  │  │  read(handle)                            │  │
  │  │  write(handle)                           │  │
  │  │  ...                                     │  │
  │  └──────────────────────────────────────────┘  │
  │                                                │
  │  finally: close(handle)                        │
  └────────────────────────────────────────────────┘
```

Niche: database connections, file handles, MCP tool bundles,
lock acquire/release, transaction scopes, disposable patterns.
