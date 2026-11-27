# Stable public API header manifest.  GENERATED -- do not edit by hand.
#
#   python3 scripts/check_public_headers.py --write-install-manifest
#
# This is the *transitive include closure* of the published C++/C API
# (zvec/db/**, zvec/c_api.h, zvec/export.h): the complete set of headers an SDK
# consumer needs to compile against libzvec and libzvec_c_api, and therefore the
# set CMake always installs.  scripts/check_public_headers.py recomputes the
# closure from the sources and fails when this list drifts.
#
# Paths are relative to src/include/.  The closure reaches no engine header
# (rule 6): the three zvec/detail/ entries are the only implementation detail it
# needs, and they deliberately live outside zvec/core, zvec/ailego and
# zvec/turbo.  This list is the complete installed header set: the engine
# directories are build-tree only, and no package carries them.
set(ZVEC_STABLE_PUBLIC_HEADERS
    zvec/c_api.h
    zvec/export.h

    zvec/db/collection.h
    zvec/db/config.h
    zvec/db/doc.h
    zvec/db/doc_iterator.h
    zvec/db/index_params.h
    zvec/db/options.h
    zvec/db/query.h
    zvec/db/query_params.h
    zvec/db/reranker.h
    zvec/db/schema.h
    zvec/db/stats.h
    zvec/db/status.h
    zvec/db/type.h

    zvec/detail/expected.hpp
    zvec/detail/float_helper.h
    zvec/detail/index_defaults.h
)
