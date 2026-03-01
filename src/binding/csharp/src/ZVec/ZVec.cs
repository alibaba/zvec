// Copyright 2025-present the zvec project
using System;
using System.Runtime.InteropServices;

namespace ZVec
{
    public class ZVecException : Exception
    {
        public int Code { get; }
        public ZVecException(int code, string message) : base(message) { Code = code; }
    }

    internal static class StatusHelper
    {
        internal static void Check(IntPtr status)
        {
            if (status == IntPtr.Zero) return;
            int code = NativeMethods.zvec_status_code(status);
            string msg = Marshal.PtrToStringUTF8(NativeMethods.zvec_status_message(status)) ?? "Unknown error";
            NativeMethods.zvec_status_destroy(status);
            throw new ZVecException(code, msg);
        }
    }

    public class Schema : IDisposable
    {
        internal IntPtr Handle { get; private set; }

        public Schema(string name)
        {
            Handle = NativeMethods.zvec_schema_create(name);
        }

        public void AddField(string name, DataType type, uint dimension = 0)
        {
            StatusHelper.Check(NativeMethods.zvec_schema_add_field(Handle, name, (int)type, dimension));
        }

        public void Dispose()
        {
            if (Handle != IntPtr.Zero) { NativeMethods.zvec_schema_destroy(Handle); Handle = IntPtr.Zero; }
        }
    }

    public class Doc : IDisposable
    {
        internal IntPtr Handle { get; private set; }
        private bool _owns;

        public Doc()
        {
            Handle = NativeMethods.zvec_doc_create();
            _owns = true;
        }

        internal Doc(IntPtr handle)
        {
            Handle = handle;
            _owns = true;
        }

        public void SetPK(string pk) => NativeMethods.zvec_doc_set_pk(Handle, pk);

        public string PK()
        {
            IntPtr ptr = NativeMethods.zvec_doc_pk(Handle);
            return Marshal.PtrToStringUTF8(ptr) ?? "";
        }

        public void SetString(string field, string value) =>
            StatusHelper.Check(NativeMethods.zvec_doc_set_string(Handle, field, value));

        public void SetInt32(string field, int value) =>
            StatusHelper.Check(NativeMethods.zvec_doc_set_int32(Handle, field, value));

        public void SetVector(string field, float[] vector) =>
            StatusHelper.Check(NativeMethods.zvec_doc_set_float_vector(Handle, field, vector, (uint)vector.Length));

        public float Score() => NativeMethods.zvec_doc_score(Handle);

        public void Dispose()
        {
            if (_owns && Handle != IntPtr.Zero) { NativeMethods.zvec_doc_destroy(Handle); Handle = IntPtr.Zero; }
        }
    }

    public class Collection : IDisposable
    {
        private IntPtr _handle;

        public Collection(string path, Schema schema)
        {
            StatusHelper.Check(NativeMethods.zvec_collection_create_and_open(path, schema.Handle, out _handle));
        }

        public Collection(string path)
        {
            StatusHelper.Check(NativeMethods.zvec_collection_open(path, out _handle));
        }

        public void Upsert(Doc[] docs)
        {
            IntPtr[] ptrs = Array.ConvertAll(docs, d => d.Handle);
            StatusHelper.Check(NativeMethods.zvec_collection_upsert(_handle, ptrs, (nuint)ptrs.Length));
        }

        public Doc[] Fetch(string[] pks)
        {
            IntPtr[] pkPtrs = new IntPtr[pks.Length];
            for (int i = 0; i < pks.Length; i++)
                pkPtrs[i] = Marshal.StringToCoTaskMemUTF8(pks[i]);

            try
            {
                StatusHelper.Check(NativeMethods.zvec_collection_fetch(_handle, pkPtrs, (nuint)pks.Length, out IntPtr list));
                nuint size = NativeMethods.zvec_doc_list_size(list);
                Doc[] results = new Doc[(int)size];
                for (nuint i = 0; i < size; i++)
                    results[(int)i] = new Doc(NativeMethods.zvec_doc_list_get(list, i));
                NativeMethods.zvec_doc_list_destroy(list);
                return results;
            }
            finally
            {
                foreach (var p in pkPtrs) Marshal.FreeCoTaskMem(p);
            }
        }

        public void Flush() => StatusHelper.Check(NativeMethods.zvec_collection_flush(_handle));

        public ulong Stats()
        {
            StatusHelper.Check(NativeMethods.zvec_collection_get_stats(_handle, out IntPtr stats));
            ulong count = NativeMethods.zvec_stats_total_docs(stats);
            NativeMethods.zvec_stats_destroy(stats);
            return count;
        }

        public void DestroyPhysical() => StatusHelper.Check(NativeMethods.zvec_collection_destroy_physical(_handle));

        public void Dispose()
        {
            if (_handle != IntPtr.Zero) { NativeMethods.zvec_collection_destroy(_handle); _handle = IntPtr.Zero; }
        }
    }
}
