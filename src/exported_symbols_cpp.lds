{
  global:
    extern "C++" {
      /* Unquoted = glob pattern matching all demangled zvec:: symbols.
         Do NOT add a quoted "zvec::*" entry: quotes mean literal match,
         and lld (Android NDK) fails with --no-undefined-version when a
         literal entry matches no symbol. */
      zvec::*;
    };
    zvec_*;
    /* vtable / typeinfo / VTT / guard variables of zvec:: classes.
       Their demangled names start with "vtable for" etc., so the
       zvec::* patterns above do not match them; without these the
       consumers fail with "undefined reference to vtable". Mirrors
       __ZTVN4zvec* etc. in exported_symbols_cpp.txt on macOS. */
    _ZTVN4zvec*;
    _ZTIN4zvec*;
    _ZTSN4zvec*;
    _ZTTN4zvec*;
    _ZGVN4zvec*;
    /* Function-local singleton guards must coalesce with their exported
       objects across the executable/DSO boundary, just as on macOS. */
    _ZGVZN4zvec*;
    _ZGVZNK4zvec*;
  local:
    *;
};
