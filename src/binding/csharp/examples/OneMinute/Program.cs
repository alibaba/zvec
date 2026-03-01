// Copyright 2025-present the zvec project
using System;

namespace ZVec.Examples
{
    class OneMinute
    {
        static void Main(string[] args)
        {
            string path = "/tmp/zvec_csharp_example";
            if (System.IO.Directory.Exists(path))
                System.IO.Directory.Delete(path, true);

            // 1. Create Schema
            using var schema = new Schema("test_collection");
            schema.AddField("vector", DataType.VectorFP32, 128);
            schema.AddField("age", DataType.Int32);

            // 2. Create and Open Collection
            using var collection = new Collection(path, schema);

            // 3. Upsert Documents
            using var doc = new Doc();
            doc.SetPK("user_1");
            doc.SetVector("vector", new float[128]);
            doc.SetInt32("age", 30);
            collection.Upsert(new[] { doc });
            collection.Flush();

            // 4. Get Stats
            Console.WriteLine($"Total docs: {collection.Stats()}");

            // 5. Fetch Document
            var docs = collection.Fetch(new[] { "user_1" });
            foreach (var d in docs)
            {
                Console.WriteLine($"Fetched PK: {d.PK()}");
                d.Dispose();
            }
        }
    }
}
