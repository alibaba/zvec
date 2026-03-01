// Copyright 2025-present the zvec project
using System;
using System.Runtime.InteropServices;

namespace ZVec
{
    internal static class NativeMethods
    {
        private const string LibName = "zvec_c";

        // --- Status API ---
        [DllImport(LibName)] internal static extern int zvec_status_code(IntPtr status);
        [DllImport(LibName)] internal static extern IntPtr zvec_status_message(IntPtr status);
        [DllImport(LibName)] internal static extern void zvec_status_destroy(IntPtr status);

        // --- Schema API ---
        [DllImport(LibName)] internal static extern IntPtr zvec_schema_create([MarshalAs(UnmanagedType.LPUTF8Str)] string name);
        [DllImport(LibName)] internal static extern void zvec_schema_destroy(IntPtr schema);
        [DllImport(LibName)] internal static extern IntPtr zvec_schema_add_field(
            IntPtr schema, [MarshalAs(UnmanagedType.LPUTF8Str)] string name,
            int type, uint dimension);

        // --- Doc API ---
        [DllImport(LibName)] internal static extern IntPtr zvec_doc_create();
        [DllImport(LibName)] internal static extern void zvec_doc_destroy(IntPtr doc);
        [DllImport(LibName)] internal static extern void zvec_doc_set_pk(IntPtr doc, [MarshalAs(UnmanagedType.LPUTF8Str)] string pk);
        [DllImport(LibName)] internal static extern IntPtr zvec_doc_pk(IntPtr doc);
        [DllImport(LibName)] internal static extern void zvec_doc_set_score(IntPtr doc, float score);
        [DllImport(LibName)] internal static extern float zvec_doc_score(IntPtr doc);
        [DllImport(LibName)] internal static extern IntPtr zvec_doc_set_string(
            IntPtr doc, [MarshalAs(UnmanagedType.LPUTF8Str)] string field,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string value);
        [DllImport(LibName)] internal static extern IntPtr zvec_doc_set_int32(
            IntPtr doc, [MarshalAs(UnmanagedType.LPUTF8Str)] string field, int value);
        [DllImport(LibName)] internal static extern IntPtr zvec_doc_set_float(
            IntPtr doc, [MarshalAs(UnmanagedType.LPUTF8Str)] string field, float value);
        [DllImport(LibName)] internal static extern IntPtr zvec_doc_set_float_vector(
            IntPtr doc, [MarshalAs(UnmanagedType.LPUTF8Str)] string field,
            float[] data, uint count);

        // --- Collection API ---
        [DllImport(LibName)] internal static extern IntPtr zvec_collection_create_and_open(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string path, IntPtr schema, out IntPtr collection);
        [DllImport(LibName)] internal static extern IntPtr zvec_collection_open(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string path, out IntPtr collection);
        [DllImport(LibName)] internal static extern void zvec_collection_destroy(IntPtr collection);
        [DllImport(LibName)] internal static extern IntPtr zvec_collection_flush(IntPtr collection);
        [DllImport(LibName)] internal static extern IntPtr zvec_collection_destroy_physical(IntPtr collection);
        [DllImport(LibName)] internal static extern IntPtr zvec_collection_get_stats(IntPtr collection, out IntPtr stats);

        // DML
        [DllImport(LibName)] internal static extern IntPtr zvec_collection_upsert(IntPtr collection, IntPtr[] docs, nuint count);
        [DllImport(LibName)] internal static extern IntPtr zvec_collection_insert(IntPtr collection, IntPtr[] docs, nuint count);
        [DllImport(LibName)] internal static extern IntPtr zvec_collection_update(IntPtr collection, IntPtr[] docs, nuint count);
        [DllImport(LibName)] internal static extern IntPtr zvec_collection_delete(IntPtr collection, IntPtr[] pks, nuint count);

        // DQL
        [DllImport(LibName)] internal static extern IntPtr zvec_collection_fetch(
            IntPtr collection, IntPtr[] pks, nuint count, out IntPtr results);
        [DllImport(LibName)] internal static extern IntPtr zvec_collection_query(
            IntPtr collection, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName,
            float[] vector, uint count, int topk, out IntPtr results);

        // --- Doc List API ---
        [DllImport(LibName)] internal static extern nuint zvec_doc_list_size(IntPtr list);
        [DllImport(LibName)] internal static extern IntPtr zvec_doc_list_get(IntPtr list, nuint index);
        [DllImport(LibName)] internal static extern void zvec_doc_list_destroy(IntPtr list);

        // --- Stats API ---
        [DllImport(LibName)] internal static extern ulong zvec_stats_total_docs(IntPtr stats);
        [DllImport(LibName)] internal static extern void zvec_stats_destroy(IntPtr stats);
    }
}
