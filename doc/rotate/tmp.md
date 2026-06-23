### linux-x64-clang:

segment_helper_test最后一个run:

[ INFO 2026-06-22 16:04:45 139636402713984 local_wal_file.cc:118] Wal close success wal_path_[./test_collection/2/0.wal] 
[ INFO 2026-06-22 16:04:45 139636402713984 local_wal_file.cc:129] Wal remove success. wal_path_[./test_collection/2/0.wal] 
[ INFO 2026-06-22 16:04:45 139636402713984 segment.cc:2313] WAL cleaned up: segment[2]
FhtKacRotator is selected
[ INFO 2026-06-22 16:04:45 139636402713984 rabitq_converter.cc:107] RabitqConverter initialized: dim=128, padded_dim=128, num_clusters=256, ex_bits=6, rotator_type=1[] sample_count[0]
[ INFO 2026-06-22 16:04:45 139636402713984 rabitq_converter.cc:140] Training with 300 vectors from 300 of holder
[ INFO 2026-06-22 16:04:45 139636402713984 rabitq_converter.cc:171] Initializing KmeansCluster with meta: dim=128, data_type=2, metric=InnerProduct
[ INFO 2026-06-22 16:04:45 139636402713984 rabitq_converter.cc:217] Training completed: 256 centroids, cost 43 ms
[ INFO 2026-06-22 16:04:45 139636402713984 rabitq_converter.cc:263] Dump completed: 262240 bytes, cost 0 ms
FhtKacRotator is selected
[ INFO 2026-06-22 16:04:45 139636402713984 rabitq_reformer.cc:237] Rabitq reformer load done. dimension=128, padded_dim=128, ex_bits=6, num_clusters=256, size_bin_data=28, size_ex_data=104 rotator_type=1
[ INFO 2026-06-22 16:04:45 139636402713984 hnsw_rabitq_streamer.cc:282] HnswRabitqStreamer open
[ INFO 2026-06-22 16:04:45 139636402713984 rabitq_reformer.cc:465] RabitqReformer dump to storage completed: 262240 bytes
[ INFO 2026-06-22 16:04:45 139636402713984 hnsw_rabitq_streamer.cc:312] Dump reformer success.
[ INFO 2026-06-22 16:04:45 139636402713984 hnsw_rabitq_query_algorithm.cc:39] Create query algorithm. num_clusters=256 ex_bits=6 padded_dim=128

collection_test最后一个run：

[ RUN      ] CollectionTest.Feature_Optimize_Repeated
FhtKacRotator is selected
FhtKacRotator is selected

### windows:
=================================== ERRORS ====================================
_____ ERROR collecting python/tests/detail/test_collection_concurrency.py _____
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\detail\test_collection_concurrency.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\detail\test_collection_concurrency.py:19: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
___ ERROR collecting python/tests/detail/test_collection_create_and_open.py ___
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\detail\test_collection_create_and_open.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\detail\test_collection_create_and_open.py:17: in <module>
    from distance_helper import *
python\tests\detail\distance_helper.py:5: in <module>
    from zvec import (
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
_________ ERROR collecting python/tests/detail/test_collection_ddl.py _________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\detail\test_collection_ddl.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\detail\test_collection_ddl.py:15: in <module>
    from distance_helper import *
python\tests\detail\distance_helper.py:5: in <module>
    from zvec import (
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
_________ ERROR collecting python/tests/detail/test_collection_dml.py _________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\detail\test_collection_dml.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\detail\test_collection_dml.py:5: in <module>
    from zvec import (
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
_________ ERROR collecting python/tests/detail/test_collection_dql.py _________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\detail\test_collection_dql.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\detail\test_collection_dql.py:16: in <module>
    from distance_helper import *
python\tests\detail\distance_helper.py:5: in <module>
    from zvec import (
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
______ ERROR collecting python/tests/detail/test_collection_exception.py ______
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\detail\test_collection_exception.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\detail\test_collection_exception.py:19: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
________ ERROR collecting python/tests/detail/test_collection_open.py _________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\detail\test_collection_open.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\detail\test_collection_open.py:19: in <module>
    from fixture_helper import *
python\tests\detail\fixture_helper.py:13: in <module>
    from zvec.typing import DataType, StatusCode, MetricType, QuantizeType
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
_______ ERROR collecting python/tests/detail/test_collection_recall.py ________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\detail\test_collection_recall.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\detail\test_collection_recall.py:17: in <module>
    from zvec.typing import DataType, StatusCode, MetricType, QuantizeType
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
___________ ERROR collecting python/tests/detail/test_db_config.py ____________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\detail\test_db_config.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\detail\test_db_config.py:22: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
______________ ERROR collecting python/tests/test_collection.py _______________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_collection.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_collection.py:17: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
__________ ERROR collecting python/tests/test_collection_diskann.py ___________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_collection_diskann.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_collection_diskann.py:56: in <module>
    import zvec  # noqa: E402
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
____________ ERROR collecting python/tests/test_collection_fts.py _____________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_collection_fts.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_collection_fts.py:24: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
_____ ERROR collecting python/tests/test_collection_fts_vector_hybrid.py ______
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_collection_fts_vector_hybrid.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_collection_fts_vector_hybrid.py:19: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
________ ERROR collecting python/tests/test_collection_hnsw_rabitq.py _________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_collection_hnsw_rabitq.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_collection_hnsw_rabitq.py:21: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
________________ ERROR collecting python/tests/test_convert.py ________________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_convert.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_convert.py:6: in <module>
    from _zvec import _Doc
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
__________________ ERROR collecting python/tests/test_doc.py __________________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_doc.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_doc.py:20: in <module>
    from _zvec import _Doc
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
_______________ ERROR collecting python/tests/test_embedding.py _______________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_embedding.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_embedding.py:22: in <module>
    from zvec.extension import (
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
_______________ ERROR collecting python/tests/test_fts_query.py _______________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_fts_query.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_fts_query.py:20: in <module>
    from zvec.model.param.query import Fts, Query
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
______________ ERROR collecting python/tests/test_gil_release.py ______________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_gil_release.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_gil_release.py:26: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
________ ERROR collecting python/tests/test_hnsw_contiguous_memory.py _________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_hnsw_contiguous_memory.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_hnsw_contiguous_memory.py:39: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
__________ ERROR collecting python/tests/test_jieba_default_dict.py ___________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_jieba_default_dict.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_jieba_default_dict.py:32: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
________________ ERROR collecting python/tests/test_params.py _________________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_params.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_params.py:22: in <module>
    from zvec import (
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
____________ ERROR collecting python/tests/test_query_executor.py _____________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_query_executor.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_query_executor.py:21: in <module>
    from _zvec.param import _SearchQuery
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
_______________ ERROR collecting python/tests/test_reranker.py ________________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_reranker.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_reranker.py:20: in <module>
    from zvec import Doc, MetricType, VectorSchema, DataType, FlatIndexParam
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
________________ ERROR collecting python/tests/test_schema.py _________________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_schema.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_schema.py:17: in <module>
    from zvec import (
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
________________ ERROR collecting python/tests/test_typing.py _________________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_typing.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_typing.py:17: in <module>
    from zvec import (
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
_________________ ERROR collecting python/tests/test_util.py __________________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_util.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_util.py:19: in <module>
    from zvec import require_module
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
________________ ERROR collecting python/tests/test_vamana.py _________________
ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_vamana.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_vamana.py:39: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
=========================== short test summary info ===========================
ERROR python/tests/detail/test_collection_concurrency.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\detail\test_collection_concurrency.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\detail\test_collection_concurrency.py:19: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/detail/test_collection_create_and_open.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\detail\test_collection_create_and_open.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\detail\test_collection_create_and_open.py:17: in <module>
    from distance_helper import *
python\tests\detail\distance_helper.py:5: in <module>
    from zvec import (
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/detail/test_collection_ddl.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\detail\test_collection_ddl.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\detail\test_collection_ddl.py:15: in <module>
    from distance_helper import *
python\tests\detail\distance_helper.py:5: in <module>
    from zvec import (
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/detail/test_collection_dml.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\detail\test_collection_dml.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\detail\test_collection_dml.py:5: in <module>
    from zvec import (
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/detail/test_collection_dql.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\detail\test_collection_dql.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\detail\test_collection_dql.py:16: in <module>
    from distance_helper import *
python\tests\detail\distance_helper.py:5: in <module>
    from zvec import (
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/detail/test_collection_exception.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\detail\test_collection_exception.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\detail\test_collection_exception.py:19: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/detail/test_collection_open.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\detail\test_collection_open.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\detail\test_collection_open.py:19: in <module>
    from fixture_helper import *
python\tests\detail\fixture_helper.py:13: in <module>
    from zvec.typing import DataType, StatusCode, MetricType, QuantizeType
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/detail/test_collection_recall.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\detail\test_collection_recall.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\detail\test_collection_recall.py:17: in <module>
    from zvec.typing import DataType, StatusCode, MetricType, QuantizeType
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/detail/test_db_config.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\detail\test_db_config.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\detail\test_db_config.py:22: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/test_collection.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_collection.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_collection.py:17: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/test_collection_diskann.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_collection_diskann.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_collection_diskann.py:56: in <module>
    import zvec  # noqa: E402
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/test_collection_fts.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_collection_fts.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_collection_fts.py:24: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/test_collection_fts_vector_hybrid.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_collection_fts_vector_hybrid.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_collection_fts_vector_hybrid.py:19: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/test_collection_hnsw_rabitq.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_collection_hnsw_rabitq.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_collection_hnsw_rabitq.py:21: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/test_convert.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_convert.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_convert.py:6: in <module>
    from _zvec import _Doc
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/test_doc.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_doc.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_doc.py:20: in <module>
    from _zvec import _Doc
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/test_embedding.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_embedding.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_embedding.py:22: in <module>
    from zvec.extension import (
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/test_fts_query.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_fts_query.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_fts_query.py:20: in <module>
    from zvec.model.param.query import Fts, Query
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/test_gil_release.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_gil_release.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_gil_release.py:26: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/test_hnsw_contiguous_memory.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_hnsw_contiguous_memory.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_hnsw_contiguous_memory.py:39: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/test_jieba_default_dict.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_jieba_default_dict.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_jieba_default_dict.py:32: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/test_params.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_params.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_params.py:22: in <module>
    from zvec import (
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/test_query_executor.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_query_executor.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_query_executor.py:21: in <module>
    from _zvec.param import _SearchQuery
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/test_reranker.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_reranker.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_reranker.py:20: in <module>
    from zvec import Doc, MetricType, VectorSchema, DataType, FlatIndexParam
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/test_schema.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_schema.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_schema.py:17: in <module>
    from zvec import (
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/test_typing.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_typing.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_typing.py:17: in <module>
    from zvec import (
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/test_util.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_util.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_util.py:19: in <module>
    from zvec import require_module
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
ERROR python/tests/test_vamana.py - ImportError while importing test module 'D:\a\zvec\zvec\python\tests\test_vamana.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
python\tests\test_vamana.py:39: in <module>
    import zvec
C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\site-packages\zvec\__init__.py:51: in <module>
    from _zvec import (
E   ImportError: DLL load failed while importing _zvec: The specified module could not be found.
============================= 28 errors in 3.58s ==============================
Error: Process completed with exit code 1.