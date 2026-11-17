# Experimental HNSW duplicate leaves

Large groups of identical vectors should not consume navigation edges or the
representative search budget. This prototype gives each byte-identical stored
vector one navigation representative, while preserving every record's ID, key,
vector, filtering behavior, and fetch behavior.

## Enable

The option is off by default. The C++ core interface exposes it as:

```cpp
zvec::core_interface::HNSWIndexParam param;
param.dimension = 128;
param.data_type = zvec::core_interface::DataType::DT_FP32;
param.deduplicate_vectors = true;
```

`HNSWIndexParamBuilder::with_deduplicate_vectors(true)` and the serialized
`deduplicate_vectors` field are equivalent. The low-level streamer parameter is
`proxima.hnsw.streamer.deduplicate_vectors`.

This review prototype does not add a DB schema/protobuf field or Python API.
It supports dense, in-index vectors with the ordinary metric path or
`Fp32Quantizer`. External vectors, an original-vector provider, compressed
quantizers, and sparse HNSW are rejected rather than deduplicated in a different
distance space. The borrowed-vector bulk-merge optimization is bypassed for
this option; the regular merge path remains available.

## Graph and query behavior

The first inserted instance becomes the representative. Only representatives
participate in HNSW neighbor selection, reverse-edge pruning, promotion, and
navigation. A hash directory finds groups and confirms every hit by comparing
the complete stored vector bytes. Neither a hash collision nor a zero distance
is sufficient to merge vectors.

Each node has a separate pair of 32-bit IDs after its L0 navigation adjacency:

| Node | `representative` | `next` |
| --- | --- | --- |
| Representative | Its own ID | First duplicate, or invalid |
| Duplicate leaf | Representative ID | Next duplicate, or invalid |

Every leaf additionally has one ordinary L0 back edge to the representative.
The representative-to-member relation is stored as a posting list, not as an
unbounded ordinary adjacency list. Thus the relation is bidirectional without
consuming the representative's `M` navigation slots. The combined structure
exposes the leaves during result expansion; ordinary navigation adjacency alone
does not enumerate them.

Search first selects representative candidates. Before exporting records, it
expands their members into a separate result heap using the identical vector
score. Filters are checked on each member. A filtered-out representative still
qualifies as a navigation result if one of its members passes the filter.
Duplicates consume result slots, but do not consume navigation candidate slots
or incur additional distance evaluations. At most `max(topk, ef)` accepted
members per vector group need expansion. Tied record IDs remain unspecified.

Group-by queries use the existing brute-force path so that distinct business
groups sharing one vector remain visible. This preserves semantics at the cost
of group-by query performance.

## Concurrency

Node allocation and navigation updates retain their existing HNSW locks. The
directory mutex only protects group lookup/reservation and publication; distinct
representatives build concurrently. Concurrent duplicates wait for their own
representative's build to complete. A failed build publishes its error to those
waiters, including on exceptions.

Attaching a leaf uses the existing representative node's striped mutex. Search
takes that same mutex only to snapshot the posting-list head. Published leaf
links are immutable, so traversal then proceeds without holding a lock. New
members appended after that snapshot become visible on subsequent queries.

## Persistence and costs

Mutable storage flush/close/reopen persists the links with the node records.
The header records the layout option, and reopening with the wrong option is
rejected. The flag cannot be toggled for an existing index: rebuild it instead.
Readers of an enabled index must use this implementation; compatibility with
older readers is not promised. Disabled indexes retain the original format.
The legacy split-segment `dump()` export returns `IndexError_Unsupported` for an
enabled index rather than silently losing the posting links.

There are 8 additional logical bytes per node, potentially up to 32 additional
physical bytes after node alignment. The directory uses O(unique vectors)
memory, including publication state. Opening reconstructs it by scanning node
links and hashing representative vectors. This prototype retains the original
vector allocation and random upper-slot allocation for every record; it reduces
navigation work rather than implementing vector-storage compression.

Replacing an already occupied explicit ID is rejected; filling an unused ID gap
is supported. Record deletion remains the existing query-filter/rebuild model.

## Near duplicates

No epsilon or similarity threshold is introduced. Near duplicates remain normal
navigation nodes. Attaching them would require actual member distance evaluation
and a separate recall study: exact score reuse and representative routing are
only safe here for identical stored records.

## Validation

`hnsw_duplicates_test` covers identical groups larger than `M`, per-record
filtering (including a rejected representative), candidate export, navigation to
other vectors, flush/reopen/append, explicit ID gaps, near duplicates, same-group
and mixed-group concurrent insertion, overlapping construction of distinct
representatives, FP32 L2/cosine/inner-product paths, group-by fallback, default
behavior, layout mismatch rejection, and contiguous read-only queries.
`hnsw_duplicate_param_test` checks the C++ builder and serialization.
