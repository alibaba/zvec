# Manifest Wire Format

This directory holds the **implementation details** of the zvec manifest
on-disk format. The public API deliberately lives outside of it:

- Public API: `db/index/common/manifest_codec.h`
  (`ManifestData`, `ManifestCodec::Encode/Decode`; the per-message helpers
  declared there are exposed only for unit testing).
- Wire enums: `db/index/common/manifest_enum.h`
  (shared with the public `type_helper.h`).
- Consumer: `Version::Load/Save` in `db/index/common/version_manager.cc`.

Files in this directory:

- `manifest_codec.cc` — the field numbers (the `f_*` namespaces) and the
  encode/decode logic for every message.
- `pb_wire.{h,cc}` — the minimal protobuf wire-format reader/writer used by
  the codec (varint, length-delimited, fixed32/fixed64).

## Why not libprotobuf?

The manifest format **is** the protobuf wire format
(https://protobuf.dev/programming-guides/encoding/), so manifests stay
byte-compatible with the previous protobuf-based implementation. The
libprotobuf/protoc dependency was dropped to slim down the shared library;
`manifest_codec.cc` together with `manifest_enum.h` now defines the format.

## Compatibility rules

The manifest is a persisted artifact: files written by one zvec version must
remain readable by later versions, and — thanks to protobuf's skipping of
unknown fields — also by older versions.

### Adding a field (allowed at any time)

1. Pick the next unused field number in the message's `f_*` namespace in
   `manifest_codec.cc`. Field numbers are the identity of a field.
2. Encode it with the matching `Writer::Put*` call. Singular fields holding
   proto3 default values are skipped on write, so manifests written by older
   versions decode to the default automatically.
3. Read it back in the message's `Decode*` function. Unknown fields are
   skipped by readers, so older binaries keep working on newer files.
4. Extend the C++ side (`ManifestData`, index params, ...) as needed.
5. Add test coverage. The golden tests in
   `tests/db/index/common/manifest_codec_golden_test.cc` pin the exact bytes
   produced by the old protobuf implementation: adding a field must not
   change existing golden bytes (default-valued fields are not serialized),
   and new fields get their own cases.

### Removing a field (allowed, with care)

1. Stop encoding the field. Existing manifests still decode: the field simply
   falls back to its default value on load.
2. Keep the field number in the `f_*` namespace, marked as reserved with a
   comment. **Never reuse a retired field number** for a different meaning —
   old manifest files would silently decode garbage into the new field.
3. If the removed field carried information that is still required, migrate
   it to a new field number first.

### Renaming a field

Only field numbers go over the wire; names may change freely.

### Changing enums

Enums are encoded as their numeric values (`manifest_enum.h` mirrors the old
proto numbers). Existing numbers must never change; add new values with new
numbers. Unknown values read from disk are preserved as-is and mapped to
`UNDEFINED` by the CodeBooks in `type_helper.h`.

### Writer constraint

`pbwire::Writer` does not sort fields. Each message's encode function must
write fields in ascending field-number order to produce canonical bytes.
