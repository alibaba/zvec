# zvec:db — WIT/WASM Bindings

WebAssembly Interface Types (WIT) definition and guest-side Rust implementation for [zvec](https://zvec.org).

## Package

```
package zvec:db@0.1.0
```

## WIT Interface

The `zvec:db` WIT world exposes the full zvec API as WASM Component Model resources:

| Resource      | Operations |
|--------------|------------|
| `schema`     | `constructor(name)`, `add-field(name, type, dim)` |
| `doc`        | `set-pk`, `pk`, `set-vector`, `set-string`, `set-int32`, `score` |
| `collection` | `create`, `open`, `upsert`, `insert`, `update`, `delete`, `fetch`, `query`, `flush`, `stats`, `destroy-physical` |

## Guest-side Rust Crate

The `guest-rust/` directory contains a Rust crate that implements the `zvec:db` world for WASM components. Build with:

```bash
cd guest-rust
cargo build --target wasm32-wasip1 --release
```

## Using in Your WASM Component

```wit
// your-world.wit
package my:app@0.1.0;

world my-app {
    import zvec:db/types;
}
```

## Publishing

The WIT package can be published to a [warg](https://warg.io) registry:

```bash
warg publish --name zvec:db --version 0.1.0
```

## License

Apache-2.0
