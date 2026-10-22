#!/bin/sh
# Will this binary run on RHEL / CentOS 7?  Prints the highest GLIBC_x.y, GLIBCXX_x.y.z and CXXABI_x.y.z versions it
# asks the system's libraries for, and exits 1 when one is above what RHEL 7 has (glibc 2.17, libstdc++ of GCC 4.8.5:
# GLIBCXX 3.4.19, CXXABI 1.3.7).
#
#     tools/check_glibc_symbols.sh build-rhel7/strata build-rhel7/libstrata_kq.so
#     tools/check_glibc_symbols.sh build/strata --max-glibc 2.28            # RHEL 8
#
# A binary built on RHEL 7 with devtoolset needs nothing above these (devtoolset links the newer parts of libstdc++ in
# statically); one built on a newer system and copied over almost never passes, and has to be built on RHEL 7 (or in
# docker/rhel7).  It reads the dynamic symbol table, so it needs `objdump` (or `nm`) from binutils.
set -eu
MAX_GLIBC=2.17
MAX_GLIBCXX=3.4.19
MAX_CXXABI=1.3.7
FILES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --max-glibc) MAX_GLIBC="$2"; shift 2 ;;
    --max-glibcxx) MAX_GLIBCXX="$2"; shift 2 ;;
    --max-cxxabi) MAX_CXXABI="$2"; shift 2 ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "check_glibc_symbols: unknown option $1" >&2; exit 2 ;;
    *) FILES="$FILES $1"; shift ;;
  esac
done
[ -n "$FILES" ] || { echo "usage: check_glibc_symbols.sh BINARY... [--max-glibc X.Y] [--max-glibcxx X.Y.Z] [--max-cxxabi X.Y.Z]" >&2; exit 2; }

# the larger of two dotted versions
newer() { printf '%s\n%s\n' "$1" "$2" | sort -t. -k1,1n -k2,2n -k3,3n | tail -1; }

status=0
for BIN in $FILES; do
  [ -f "$BIN" ] || { echo "check_glibc_symbols: $BIN: no such file" >&2; exit 2; }
  if command -v objdump >/dev/null 2>&1; then syms=$(objdump -T "$BIN" 2>/dev/null || true)
  elif command -v nm >/dev/null 2>&1; then syms=$(nm -D --with-symbol-versions "$BIN" 2>/dev/null || true)
  else echo "check_glibc_symbols: objdump (binutils) is needed" >&2; exit 2; fi
  line="$BIN needs:"
  for fam in GLIBC:$MAX_GLIBC GLIBCXX:$MAX_GLIBCXX CXXABI:$MAX_CXXABI; do
    name=${fam%%:*}; max=${fam#*:}
    have=$(printf '%s\n' "$syms" | grep -o "$name"'_[0-9][0-9.]*' | sed "s/^${name}_//" | sort -t. -k1,1n -k2,2n -k3,3n | tail -1 || true)
    line="$line $name ${have:-none}"
    if [ -n "$have" ] && [ "$(newer "$have" "$max")" != "$max" ]; then
      echo "$line" >&2
      echo "check_glibc_symbols: ${name}_$have is newer than ${name}_$max; the symbols that ask for it:" >&2
      printf '%s\n' "$syms" | grep "${name}_$have" | head -5 >&2
      status=1
      line="$BIN needs:"
    fi
  done
  [ "$line" = "$BIN needs:" ] || echo "$line (ok)"
done
exit $status
