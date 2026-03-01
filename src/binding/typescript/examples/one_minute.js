// Copyright 2025-present the zvec project
// One-minute example for zvec Node.js binding
const { Schema, Doc, Collection, DataType } = require('../lib/index');
const fs = require('fs');
const path = '/tmp/zvec_node_example';

// Clean up previous run
fs.rmSync(path, { recursive: true, force: true });

// 1. Create Schema
const schema = new Schema('test_collection');
schema.addField('vector', DataType.VectorFP32, 128);
schema.addField('age', DataType.Int32);

// 2. Create and Open Collection
const collection = new Collection(path, schema);

// 3. Upsert Documents
const doc = new Doc();
doc.setPK('user_1');
doc.setVector('vector', new Float32Array(128).fill(0.1));
doc.setInt32('age', 30);

collection.upsert([doc]);
collection.flush();

// 4. Get Stats
console.log('Total docs:', collection.stats());

// 5. Fetch Document
const docs = collection.fetch(['user_1']);
for (const d of docs) {
  console.log('Fetched PK:', d.pk());
}
