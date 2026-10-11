# Dynarec Suspend, Polling, Memory-Fault, and Invalidation Notes

This note collects the current design ideas discussed for `ryurs-dynarec` around
thread suspend, execution-context polling, recoverable memory faults, and JIT
invalidation. It is intentionally design-oriented rather than implementation-
ready code.

## 1. Goals

We want to support these behaviors without forcing precise mid-block guest state
materialization on every asynchronous event:

- Suspend a running JIT thread and later resume it.
- Distinguish suspend-related polling faults from true guest memory faults.
- Handle host-page-size vs guest-page-size permission mismatches.
- Invalidate stale JIT blocks after mapping or permission changes.
- Keep the common path fast, while accepting a heavier slow path for rare events.

## 2. High-Level Split

The current discussion naturally separates into two different mechanisms:

### 2.1 Thread suspend / safepoint

This is a host/runtime concern.

- The thread may be paused for debugger-like suspend, stop-the-world, or runtime
  coordination.
- The desired behavior is often: pause now, do some host-side work, then resume
  the exact JIT continuation.
- This is different from guest-visible traps and should not necessarily force a
  one-way exit out of the current JIT block.

### 2.2 JIT invalidation

This is a code-cache correctness concern.

- When guest mappings or permissions change, stale compiled blocks must not be
  re-entered.
- Already-running stale code may need to be interrupted, but future entries are
  a separate problem.
- This should not rely on the same semantics as thread suspend.

The main conclusion is: suspend and invalidation can share some infrastructure,
but they should not be treated as the same kind of event.

## 3. Execution Context as Suspend Polling Target

One possible suspend strategy is to deliberately use the per-thread execution-
context page as the polling target.

### 3.1 Basic idea

- Protect the execution-context page with `PROT_NONE` when the runtime wants to
  suspend a thread.
- Any JIT load/store that touches that page will fault.
- The signal handler records the fault and returns to the dispatcher landing.
- The dispatcher interprets this as a suspend event rather than a guest fault.

### 3.2 Fault classification

If this strategy is used, a suspend fault can be recognized with a simple range
check:

```text
if fault_addr >= ctx_page_start && fault_addr < ctx_page_start + page_size
```

This avoids depending on a specific register slot such as `x0`.

### 3.3 Tradeoffs

Advantages:

- No separate polling-page allocation is needed.
- Any execution-context access can serve as a suspend trigger.
- Suspend is a rare slow path, so extra complexity in the uncommon path may be
  acceptable.

Disadvantages:

- The exact suspend point depends on when the block naturally touches the
  execution context.
- Later optimizer improvements may delay or move those accesses.
- Without resumable continuation, a mid-block fault is hard to recover from
  because some guest live values may still only exist in host registers or host
  stack spills.

## 4. Why Resumable Continuations Matter

If a suspend fault can happen in the middle of a block, the runtime cannot rely
on `ExecutionContext.context` alone to resume execution.

Reasons:

- Some guest register values may still live only in host registers.
- Some values may be spilled to host stack slots and not yet flushed.
- `sp`, `nzcv`, vector registers, and other state may be partially updated.

Therefore, for suspend-like events, the better model is often:

1. Save enough host continuation state to resume the current JIT execution.
2. Return to the dispatcher for host-side handling.
3. Resume the original JIT continuation when ready.

This avoids needing a precise deopt / live-value map for every possible
mid-block suspend location.

## 5. Why a Per-Thread JIT Stack Is Appealing

If the dispatcher returns to normal Rust code and later wants to resume the
faulted JIT continuation, reusing the normal thread stack is risky.

Why:

- The original JIT continuation lives on the current host stack.
- The dispatcher and Rust code may reuse the same stack before resume happens.
- That makes resuming the original continuation difficult or unsafe.

### 5.1 Proposed model

Allocate one persistent JIT stack per host thread that executes guest code.

- JIT code runs on the thread's dedicated JIT stack.
- The dispatcher and normal Rust code run on the normal host thread stack.
- Suspend saves a continuation that resumes on the JIT stack.

This is closer to a fixed-size fiber stack than a Go-style growable goroutine
stack.

### 5.2 Costs

- Extra memory per guest-running thread.
- JIT entry/exit must switch stacks.
- Guard pages and likely `sigaltstack` support should be considered.

### 5.3 Why the cost is acceptable

- Guest thread creation is already relatively heavy.
- The number of active guest threads is much smaller than typical goroutine
  counts.
- Host-to-JIT transitions are amortized by block execution and late linking.

The stack cost seems acceptable if resumable suspend is a real goal.

## 6. Soft MMU and Fault Recovery

The signal handler should do as little as possible.

Recommended behavior:

- Record `fault_addr` and minimal host-fault metadata into TLS.
- Return to the landing / dispatcher path.
- Perform all substantial logic in normal Rust code, not in the handler.

### 6.1 Classification after returning to the dispatcher

When control reaches the dispatcher, classify faults in this order:

1. **Special runtime ranges**
   - suspend polling page
   - invalidation page
   - execution-context page if it is being used as suspend polling
   - JIT stack guard page

2. **Guest reserved VA range**
   - if outside, this is likely a host bug, OOB, or unrelated crash

3. **Soft MMU guest semantics**
   - if guest semantics say the access is legal, the host fault is a false fault
     caused by protection granularity, aliasing, or lazy remap behavior
   - if guest semantics say the access is illegal, this is a true guest memory
     fault

### 6.2 Important distinction

Once a signal already happened, the host mapping is obviously not permitting the
access. The meaningful check is not the host page permission, but the guest
permission recorded by the soft MMU.

## 7. Host Page Larger Than Guest Page

This is a core problem for fastmem.

Example:

- guest pages are 4 KiB
- host pages are 16 KiB
- guest page A is `NONE`
- guest page B is `WRITE`
- A and B live inside the same host page

If the host page is mapped too permissively, accesses to A may be incorrectly
allowed. If the host page is protected too strictly, accesses to B may cause
false faults.

### 7.1 Conservative first rule

For a host page that covers multiple guest pages:

- if any covered guest page is `NONE`, treat the whole host page as non-fastmem
  and protect it conservatively

This prevents guest-illegal accesses from being silently allowed.

### 7.2 Better framing

It is useful to think in terms of host-page categories:

- uniform fastmem-safe page
- uniform truly inaccessible page
- mixed-permission page

Mixed-permission pages should not use direct fastmem blindly.

## 8. Recovering From Granularity-Mismatch Faults

Suppose the dispatcher proves that a faulted access is guest-legal, and the
fault happened only because host protection was too coarse.

### 8.1 What not to do blindly

Do not immediately switch to interpreter from the middle of the block unless
there is a precise deopt / live-state map.

Reason:

- guest live values may still exist only in host registers or host stack spills
- the current `ExecutionContext.context` may not represent full architectural
  state yet

### 8.2 Quarantine-style recovery idea

A more realistic approach is:

1. Classify the event as a fastmem granularity mismatch.
2. Mark the current block as needing non-fastmem recompilation.
3. Prevent the current block from directly chaining out to other blocks.
4. Resume the current continuation in a compatibility mode until the block
   reaches a safe boundary.
5. After the block exits, retire the old block and recompile a no-fastmem / mem-
   helper version.

### 8.3 Important caution about alias-based resume

Temporarily switching the current thread to a more permissive alias base is only
safe if used under conservative conditions.

The danger is that the faulting access may be guest-legal, while a later access
in the same block may be guest-illegal. A permissive alias could silently allow
that later access.

Therefore, alias-resume should only be considered if at least one of these is
true:

- the alias still preserves guest-granularity access semantics
- the remaining tail of the block is known not to perform unsafe memory access
- the current block has been forced into a very conservative "get to boundary"
  mode

### 8.4 Preferred long-term outcome

The long-term goal should be:

- **permanently disable fastmem for the affected block or region**
- **not** permanently disable JIT for that block

In other words, recompile as slowmem / mem-helper JIT rather than falling back
to interpreter forever.

## 9. Why Permanent Interpreter Fallback Is Too Heavy

Current behavior already falls back to interpreter on `MemoryFault` in some
cases. That is a useful safety net, but it is too pessimistic for protection-
granularity mismatch alone.

The issue is not that the block cannot be JIT-compiled. The issue is that the
block cannot safely use direct fastmem.

A better eventual model is:

- `CompiledFastmem`
- `CompiledSlowmem`
- `Interpreted`

Granularity mismatch should preferably transition `CompiledFastmem` to
`CompiledSlowmem`, not all the way to `Interpreted`.

## 10. JIT Invalidation and Linking

JIT invalidation is separate from suspend.

### 10.1 Why slot traversal alone is not enough

The current system has both:

- late-linked / return-linked outgoing edges through slots
- eager direct linking using target entry pointers embedded as constants

That means:

- clearing late-link slots only breaks some incoming edges
- eager direct-linked paths can still jump directly to a stale target

### 10.2 Better target for invalidation

The more robust invalidation target is the compiled block entry itself.

Because both eager and late links ultimately target the block entry, changing
the entry behavior is a more complete way to invalidate the block.

### 10.3 Preferred form

Prefer entry redirection or entry patching to an invalidation dispatcher stub,
not a direct jump to the normal landing target.

Why not landing:

- the landing target belongs to the execute-block shim contract
- it expects a particular saved-register and stack-frame setup
- treating it as a generic invalidation destination is brittle across platforms

Better options:

- entry thunk that loads an atomic target pointer
- metadata-driven entry redirection
- platform-specific code patching of entry bytes, if carefully engineered

### 10.4 Relation to suspend

- suspend is about pausing the currently running thread
- invalidation is about preventing future re-entry into stale code

They may cooperate, but they are not the same mechanism.

## 11. Dirty Flush as a First Correctness Strategy

For `ryurs`, a strong candidate first implementation is to use dirty-state flushes
as the boundary between speculative native fastmem and precise guest-visible
fault handling.

### 11.1 Core idea

- Compile a `Fastmem` block that performs direct native memory access.
- Before a memory access that may need precise recovery, flush enough dirty
  architectural state to `ExecutionContext.context`.
- If the access faults, do a one-way exit to the dispatcher rather than trying to
  resume inside the JIT block.
- If the fault is a recoverable fastmem mismatch rather than a true guest fault,
  recompile the block as a slower memory-access variant.

### 11.2 What must be flushed

This is not necessarily a full context writeback. The minimum set should include
whatever is required for precise guest-visible recovery:

- dirty GPRs
- dirty `sp`
- pending `nzcv`
- dirty vector / FP state when relevant
- precise current guest `pc`

### 11.3 Why this is attractive

- It avoids requiring host-continuation resume for memory faults.
- It avoids relying on permissive alias-based continuation.
- It fits naturally with the existing `JitRegCache` dirty-state model.
- It is correctness-first and easier to reason about than mid-block deopt.

### 11.4 Important correctness rule

The faultable memory operation itself must still obey guest exception ordering.
The runtime must not flush guest-visible writeback or result state that should
only become visible if the current memory operation succeeds.

## 12. PTE-Checked Blocks as a Better Slow Path

For many cases, the best fallback is not a full helper call and not permanent
interpreter fallback, but a block recompiled to use guest-page PTE checks.

### 12.1 Basic structure

For a guest virtual address `addr`:

```text
page = addr >> 12
pte = pte_table[page]

if !pte_allows_access(pte, access_kind):
    goto slow_path

host_ptr = decode_host_pointer(pte, addr)
access [host_ptr]
```

This turns guest-page permissions and backing selection into normal runtime data
rather than depending on host fault behavior alone.

### 12.2 PTE versus helper

Compared with always calling a memory helper, a PTE-checked block is usually
much cheaper:

- one table load
- a few masks/branches
- direct native memory access on success

This makes a `PteChecked` tier a strong candidate between `Fastmem` and
`Helper/Interpreted`.

### 12.3 PTE contents

A practical PTE can use one `u64` per guest `4 KiB` page. With an `8 GiB`
address space, this means:

- `8 GiB / 4 KiB = 2,097,152` guest pages
- `2,097,152 * 8 bytes = 16 MiB` for the table

One useful encoding is:

- high bits: host page base or pointer delta
- low aligned bits: flags (`R`, `W`, `X`, `Tracked`, `Helper`, `Invalid`, etc.)

PTEs do not inherently require alias mappings, but alias/mirror mappings become
useful when different guest `4 KiB` pages inside the same host page need
different effective permissions.

## 13. Selective Dirty Flush Instead of Every Access

PTE-based access does not eliminate the need for precise guest state. However,
it moves the dirty flush to the cases that truly need it.

### 13.1 Better pattern

- first do a PTE/permission check
- if access is normal mapped RAM, continue without dirty flush
- only on slow-path conditions:
  - dirty flush
  - record precise guest `pc`
  - call helper or return to dispatcher

### 13.2 Why this is attractive

- ordinary legal RAM accesses avoid repeated dirty flush cost
- true guest faults can still see precise guest architectural state
- the same slow path can handle tracking, MMIO, invalid access, and mixed-page
  behavior

### 13.3 Dirty flush versus per-access PTE lookup

There is no single universally faster option.

- `DirtyFlushEveryAccess` can be very cheap when the dirty set is tiny.
- `PteLookupThenSelectiveFlush` has a fixed per-access lookup cost, but avoids
  flushing state on accesses that never leave the fast path.

For `ryurs`, the likely practical progression is:

1. correctness-first dirty flush on likely-faultable accesses
2. PTE-checked recompilation for blocks proven unsafe for plain fastmem
3. later optimization to reduce flush frequency and cache PTE-derived results

## 14. False Positives Are Fine, False Negatives Are Not

Automatic policy selection does not need to be perfect, but its errors must be
one-sided.

- false positive: choose a slower path even though fastmem would have worked
  - acceptable; only loses performance
- false negative: allow a fast path when guest semantics required a fault or a
  slower path
  - not acceptable; this is a correctness bug

This means heuristic selection can be conservative without needing formal proof
of optimality.

## 15. Per-Access PTE Visibility Model

One workable synchronization model is to require that every real guest memory
access re-reads the relevant PTE with acquire semantics.

### 15.1 Consequences

- No block-local cached page base is required for correctness.
- Mapping/permission updates become visible to future accesses after the thread
  crosses a safepoint boundary.
- A safepoint protocol only needs to guarantee that threads have crossed a block
  boundary since the latest PTE update before the modifying syscall returns.

### 15.2 What this model does not support cheaply

This model is not strong enough to safely support:

- block-hoisted page safety assumptions
- block-level merged checks replacing later PTE loads
- cached host-view pointers surviving across safepoints

Those require a stronger invalidation/acknowledgement protocol.

## 16. Small Per-Thread Map Cache

Instead of trying to merge all checks for a block, a simpler and more robust
optimization is a tiny per-thread cache of recently used guest page to PTE
translations.

### 16.1 Proposed structure

Each `ExecutionContext` can hold a small cache, for example 4 or 8 entries:

- guest page number tag
- cached PTE value

On access:

1. derive guest page number from `addr`
2. probe the local cache
3. on miss, load the PTE from the global table and update the cache
4. validate access permissions and either continue or jump to slow path

### 16.2 Why this is preferable to static merged checks

- It handles dynamic addresses naturally.
- It does not assume later accesses in the block are independent from earlier
  loads.
- It avoids needing aggressive page-assumption invalidation within the block.
- It still reduces repeated PTE-table loads for hot pages.

## 17. Epoch-Based VM Visibility

For any design that wants `mmap`/`mprotect`/`unmap` to become globally visible at
well-defined boundaries, an epoch protocol is preferable to a single request bit.

### 17.1 Suggested protocol

Global state:

- `global_vm_epoch: AtomicU64`

Per thread / execution context:

- `observed_vm_epoch: AtomicU64`
- `in_jit: AtomicBool`
- local small-map cache

Writer thread (`mmap`, `mprotect`, `unmap`):

1. hold the global VM modification lock
2. update PTEs / permissions
3. `target_epoch = global_vm_epoch.fetch_add(1, AcqRel) + 1`
4. wait until each active JIT thread is either:
   - no longer in JIT, or
   - `observed_vm_epoch >= target_epoch`

JIT thread at safepoint (for example block entry / backedge):

1. `current = global_vm_epoch.load(Acquire)`
2. if `current != observed_vm_epoch`:
   - clear local page/PTE caches
   - discard any assumptions tied to the previous epoch
   - `observed_vm_epoch.store(current, Release)`

### 17.2 Why epoch is better than a single bit

- It does not lose repeated requests.
- It avoids request/ack ABA problems.
- It composes naturally with repeated mapping changes.
- It is easier to reason about together with block recompilation and page
  generation changes.

### 17.3 Polling page as an escalation path

Epoch is the normal synchronization mechanism. A polling-page interrupt can still
be kept as an escalation tool if a thread takes too long to reach a safepoint.

- normal case: wait for natural block-boundary acknowledgement
- pathological long-running block: arm polling page and force the thread back to
  dispatcher/safepoint

## 18. Static Analysis and Automatic Strategy Selection

The translator can use decode-time or lowering-time analysis to pick an initial
memory strategy, then refine that strategy using runtime feedback.

### 18.1 Initial static analysis inputs

- number of memory operations in the block
- estimated dirty-register cost before each memory operation
- whether the op has writeback, exclusives, pair semantics, or other tricky
  exception-ordering behavior
- whether the block is already known to touch risky pages/regions

### 18.2 Possible tier choices

- `Fastmem`
- `FastmemWithDirtyFlush`
- `PteChecked`
- `Helper`
- `Interpreted`

### 18.3 Runtime refinement

If a block faults in a way that proves fastmem is unsafe for that region, it can
be recompiled into a slower tier rather than remaining on the original policy.

This does not require formal verification of the heuristic itself. The key rule
is that the heuristic may be conservative, but it must not incorrectly allow a
too-fast path where guest semantics require a slower one.

## 19. Practical Direction for This Project

Given the current `ryurs` architecture, a pragmatic staged plan looks like:

### Stage 1: Suspend and fault classification

- Keep signal handlers minimal.
- Classify faults after returning to the dispatcher.
- Allow resumable continuation for suspend-like faults.
- Use a dedicated per-thread JIT stack if resumable continuation is pursued.

### Stage 2: Granularity mismatch handling

- Detect mixed-permission host pages.
- Treat them as non-fastmem-safe.
- On first false fault:
  - classify it as granularity mismatch
  - quarantine the current block
  - finish the current continuation conservatively if safe
  - recompile as no-fastmem / mem-helper block

### Stage 3: Invalidation improvements

- Stop relying only on link-slot clearing.
- Add a stale-entry mechanism that catches both eager and late links.
- Keep direct-link invalidation and block invalidation conceptually separate from
  suspend.

## 20. Summary of Current Preferred Ideas

- Resumable suspend is more realistic if each executing thread gets its own JIT
  stack.
- Execution-context page range checks are an acceptable way to recognize
  suspend-related polling faults if that page is intentionally used as the poll
  target.
- Soft MMU guest semantics, not host page permissions, should decide whether a
  fault is recoverable or a true guest memory fault.
- Host-page-larger-than-guest-page issues should be handled by marking mixed
  pages non-fastmem-safe.
- Granularity mismatch should ideally lead to slowmem/no-fastmem recompilation,
  not permanent interpreter fallback.
- A promising medium-term layout is `Fastmem -> PteChecked -> Helper`, rather
  than only `Fastmem -> Helper`.
- Per-access PTE lookup can work with a per-thread small translation cache and a
  global VM epoch, avoiding the need for stronger block-hoisted page
  assumptions.
- Dirty flush remains useful, but ideally only on the actual slow/fault path or
  on accesses that truly require precise recovery.
- JIT invalidation should target stale block entry behavior, not only outgoing
  link slots.
