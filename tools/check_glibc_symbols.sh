#!/bin/sh
# What C library does a binary need?  Prints the highest GLIBC_x.y, GLIBCXX_x.y.z and CXXABI_x.y.z versions it asks
# for, and exits 1 when GLIBC is above --max (default 2.17: RHEL / CentOS 7).
#
#     tools/check_glibc_symbols.sh build-q35/strata-q35
#     tools/check_glibc_symbols.sh build-q35/strata-q35 --max 2.28      # RHEL 8
#
# A binary built on RHEL 7 with devtoolset needs nothing above 2.17 (the build links libstdc++ statically, so GLIBCXX
# lines should be empty); one built on a newer system and copied to RHEL 7 usually fails here, and has to be built there.
set -eu
BIN=${1:-}
MAX=2.17
[ -n "$BIN" ] && [ -f "$BIN" ] || { echo "usage: check_glibc_symbols.sh BINARY [--max X.Y]" >&2; exit 2; }
shift
while [ $# -gt 0 ]; do
  case "$1" in --max) MAX="$2"; shift 2 ;; *) echo "unknown option $1" >&2; exit 2 ;; esac
done

if command -v objdump >/dev/null 2>&1; then syms=$(objdump -T "$BIN" 2>/dev/null)
elif command -v nm >/dev/null 2>&1; then syms=$(nm -D --with-symbol-versions "$BIN" 2>/dev/null)
else echo "check_glibc_symbols: objdump (binutils) is needed" >&2; exit 2; fi

highest() {   # highest NAME: the largest version of that family, version-sorted
  printf '%s\n' "$syms" | grep -o "$1_[0-9][0-9.]*" | sed "s/^$1_//" | sort -t. -k1,1n -k2,2n -k3,3n | tail -1
}
glibc=$(highest GLIBC); glibcxx=$(highest GLIBCXX); cxxabi=$(highest CXXABI)
echo "$BIN needs: GLIBC ${glibc:-none}, GLIBCXX ${glibcxx:-none}, CXXABI ${cxxabi:-none}"

if [ -n "$glibc" ]; then
  worst=$(printf '%s\n%s\n' "$glibc" "$MAX" | sort -t. -k1,1n -k2,2n -k3,3n | tail -1)
  if [ "$worst" != "$MAX" ]; then
    echo "check_glibc_symbols: GLIBC_$glibc is newer than $MAX; these symbols are the cause:" >&2
    printf '%s\n' "$syms" | grep "GLIBC_$glibc" | head -5 >&2
    exit 1
  fi
fi
[ -z "$glibcxx" ] || echo "note: libstdc++ is a shared dependency of this binary (GLIBCXX_$glibcxx); link it statically on the build machine" >&2
exit 0
