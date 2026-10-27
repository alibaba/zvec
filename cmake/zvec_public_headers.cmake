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
# Paths are relative to src/include/.  The zvec/core and zvec/ailego entries are
# unavoidable implementation detail reached from the stable headers; they install
# regardless of ZVEC_INSTALL_ENGINE_HEADERS, which only controls the *remaining*
# advanced engine headers.
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

    zvec/core/interface/constants.h

    zvec/ailego/pattern/expected.hpp
    zvec/ailego/utility/float_helper.h
)
