// Copyright 2025-present the zvec project
using System;
using System.IO;
using System.Collections.Generic;
using Xunit;

namespace ZVec.Tests
{
    public class ZVecTests : IDisposable
    {
        private readonly List<string> _dirs = new();

        private string TestDir(string name)
        {
            string dir = $"/tmp/zvec_csharp_test_{name}";
            if (Directory.Exists(dir)) Directory.Delete(dir, true);
            _dirs.Add(dir);
            return dir;
        }

        public void Dispose()
        {
            foreach (var dir in _dirs)
                if (Directory.Exists(dir)) Directory.Delete(dir, true);
        }

        [Fact]
        public void TestSchemaCreate()
        {
            using var schema = new Schema("test_schema");
            schema.AddField("vec", DataType.VectorFP32, 4);
            schema.AddField("name", DataType.String);
            schema.AddField("age", DataType.Int32);
        }

        [Fact]
        public void TestDocLifecycle()
        {
            using var doc = new Doc();
            doc.SetPK("pk_001");
            Assert.Equal("pk_001", doc.PK());
            doc.SetString("name", "Alice");
            doc.SetInt32("age", 30);
            doc.SetVector("vec", new float[] { 0.1f, 0.2f, 0.3f, 0.4f });
        }

        [Fact]
        public void TestCollectionCRUD()
        {
            var dir = TestDir("crud");
            using var schema = new Schema("crud_col");
            schema.AddField("vec", DataType.VectorFP32, 4);
            schema.AddField("name", DataType.String);

            using var col = new Collection(dir, schema);
            using var doc = new Doc();
            doc.SetPK("u1");
            doc.SetVector("vec", new float[] { 1, 2, 3, 4 });
            doc.SetString("name", "Bob");
            col.Upsert(new[] { doc });
            col.Flush();

            Assert.Equal(1UL, col.Stats());

            var results = col.Fetch(new[] { "u1" });
            Assert.Single(results);
            Assert.Equal("u1", results[0].PK());
            foreach (var d in results) d.Dispose();
        }

        [Fact]
        public void TestInvalidOpen()
        {
            Assert.Throws<ZVecException>(() => new Collection("/tmp/zvec_csharp_nonexistent_abc123"));
        }

        [Fact]
        public void TestBatchUpsert()
        {
            var dir = TestDir("batch");
            using var schema = new Schema("batch_col");
            schema.AddField("vec", DataType.VectorFP32, 4);

            using var col = new Collection(dir, schema);
            var docs = new Doc[100];
            for (int i = 0; i < 100; i++)
            {
                docs[i] = new Doc();
                docs[i].SetPK($"batch_{i}");
                docs[i].SetVector("vec", new float[] { i, i, i, i });
            }
            col.Upsert(docs);
            col.Flush();
            Assert.Equal(100UL, col.Stats());
            foreach (var d in docs) d.Dispose();
        }

        [Fact]
        public void TestFetchNonexistentPK()
        {
            var dir = TestDir("fetchnone");
            using var schema = new Schema("fetchnone_col");
            schema.AddField("vec", DataType.VectorFP32, 4);

            using var col = new Collection(dir, schema);
            var results = col.Fetch(new[] { "does_not_exist" });
            Assert.Empty(results);
        }
    }
}
