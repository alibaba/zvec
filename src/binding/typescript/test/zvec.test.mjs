// Copyright 2025-present the zvec project
import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { Schema, Doc, Collection, DataType } = require('../lib/index');

function testDir(name) {
  const dir = `/tmp/zvec_ts_test_${name}`;
  fs.rmSync(dir, { recursive: true, force: true });
  return dir;
}

describe('Schema', () => {
  it('should create and add fields', () => {
    const schema = new Schema('test_schema');
    schema.addField('vec', DataType.VectorFP32, 4);
    schema.addField('name', DataType.String);
    schema.addField('age', DataType.Int32);
  });
});

describe('Doc', () => {
  it('should handle full lifecycle', () => {
    const doc = new Doc();
    doc.setPK('pk_001');
    assert.equal(doc.pk(), 'pk_001');
    doc.setString('name', 'Alice');
    doc.setInt32('age', 30);
    doc.setVector('vec', new Float32Array([0.1, 0.2, 0.3, 0.4]));
  });
});

describe('Collection', () => {
  it('should handle full CRUD', () => {
    const dir = testDir('crud');
    const schema = new Schema('crud_col');
    schema.addField('vec', DataType.VectorFP32, 4);
    schema.addField('name', DataType.String);

    const col = new Collection(dir, schema);
    const doc = new Doc();
    doc.setPK('u1');
    doc.setVector('vec', new Float32Array([1.0, 2.0, 3.0, 4.0]));
    doc.setString('name', 'Bob');

    col.upsert([doc]);
    col.flush();

    assert.equal(col.stats(), 1);

    const results = col.fetch(['u1']);
    assert.equal(results.length, 1);
    assert.equal(results[0].pk(), 'u1');
  });

  it('should throw on invalid open', () => {
    assert.throws(() => new Collection('/tmp/zvec_ts_nonexistent_abc123'));
  });

  it('should handle batch upsert (100 docs)', () => {
    const dir = testDir('batch');
    const schema = new Schema('batch_col');
    schema.addField('vec', DataType.VectorFP32, 4);

    const col = new Collection(dir, schema);
    const docs = [];
    for (let i = 0; i < 100; i++) {
      const d = new Doc();
      d.setPK(`batch_${i}`);
      d.setVector('vec', new Float32Array([i, i, i, i]));
      docs.push(d);
    }
    col.upsert(docs);
    col.flush();
    assert.equal(col.stats(), 100);
  });

  it('should return empty for nonexistent PK', () => {
    const dir = testDir('fetchnone');
    const schema = new Schema('fetchnone_col');
    schema.addField('vec', DataType.VectorFP32, 4);

    const col = new Collection(dir, schema);
    const results = col.fetch(['does_not_exist']);
    assert.equal(results.length, 0);
  });
});
