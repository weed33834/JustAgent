"""Shell injection safety helpers.

A single source of truth for detecting unquoted shell metacharacters, used by
both the ``commit`` editor allowlist and the ``verify`` command allowlist so a
fix to the scanner only has to land in one place.
"""

from __future__ import annotations

# Shell control characters that are dangerous when they appear OUTSIDE quotes.
# They can split commands / introduce pipes, redirects, or command substitution
# when interpreted by a shell. Characters inside single or double quotes are
# treated as literal data (e.g. ``python -c 'a;b'`` is legitimate).
SHELL_DANGEROUS_CHARS: frozenset[str] = frozenset({";", "|", "&", "<", ">", "`", "\n", "\r"})


def contains_shell_metacharacters(command: str) -> bool:
    """Return True if ``command`` contains an unquoted shell metacharacter.

    Quote-aware scan: characters inside ``'...'`` or ``"..."`` are ignored so
    that legitimate arguments like ``python -c 'a;b'`` or ``pytest -k "a|b"``
    are accepted, while injection attempts such as ``pytest; rm -rf /`` or
    ``echo $(whoami)`` are rejected.
    """
    in_single = False
    in_double = False
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if in_single:
            if ch == "'":
                in_single = False
        elif in_double:
            if ch == "\\" and i + 1 < n:
                i += 1
            elif ch == '"':
                in_double = False
            elif ch == "`" or ch == "$" and i + 1 < n and command[i + 1] == "(":
                return True
        else:
            if ch == "'":
                in_single = True
            elif ch == '"':
                in_double = True
            elif ch in SHELL_DANGEROUS_CHARS or ch == "$" and i + 1 < n and command[i + 1] == "(":
                return True
        i += 1
    return False
