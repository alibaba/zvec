{
  global:
    extern "C++" {
      "zvec::*";
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
  local:
    *;
};
