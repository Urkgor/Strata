"""serve/jinja_lite.py - the part of Jinja2 that chat templates use, in the standard library alone.

WHY.  The model's chat template is a Jinja template (every Hugging Face model ships one), and the server renders it
for every request.  Jinja2 is a third-party package; on an old system (RHEL / CentOS 7) a server that needs nothing
but Python is worth a few hundred lines.  This module is that: the same templates, rendered to the same text.

WHAT IT IS.  A lexer, a parser and a tree-walking evaluator for the Jinja2 constructs that chat templates use:

  * `{{ expression }}`, `{% statement %}`, `{# comment #}`, with `-` and `+` whitespace control and the
    `trim_blocks` / `lstrip_blocks` settings transformers' `apply_chat_template` uses (both on here);
  * `if / elif / else`, `for` (with `else`, the `if` filter, tuple targets, `loop.*`), `break` / `continue`,
    `set` (also `set ns.x = ...` on a `namespace`, and the block form), `macro`;
  * expressions: literals, lists, tuples, dicts, attribute and item access, slices, calls with keyword arguments,
    filters, tests (`is [not] ...`), `+ - * / // % **`, `~`, comparisons, `in`, `and`, `or`, `not`, `x if c else y`;
  * Jinja's own rules for what is undefined (an undefined value prints as nothing and is false; its attributes and
    items are errors), for scoping (a `for` and a macro open a scope, an `if` does not) and for what `a.b` means
    (the attribute first, then the item).

WHAT IT IS NOT.  Template inheritance and `include` / `import`, `call` blocks, `raw`, line statements, `with`,
autoescaping and recursive loops are refused with a TemplateSyntaxError that names them.  A filter or test it does not
have is an error when it is executed, as in Jinja2 3: the template compiles, and only a path that uses it fails.

SANDBOX.  Templates come from model files and render the client's messages, so, as with Jinja2's
ImmutableSandboxedEnvironment, a template can read and cannot change: attributes whose names start with `_` are an
error, only the non-mutating methods of str, dict, list and tuple exist, and `str.format` does not.

serve/test_jinja_lite.py holds this module to Jinja2: the same output, or the same kind of error, for the repository's
templates and a collection of others, over many conversations (Jinja2 itself is needed only to run that test).
"""
import codecs
import collections.abc
import datetime
import html
import json
import re
from urllib.parse import quote as _urlquote


# ------------------------------------------------------------------------------------------------ errors
class TemplateError(Exception):
    """The base of everything this module raises on purpose."""


class TemplateSyntaxError(TemplateError):
    pass


class UndefinedError(TemplateError):
    pass


class SecurityError(TemplateError):
    pass


class TemplateRuntimeError(TemplateError):
    pass


# ------------------------------------------------------------------------------------------------ undefined
class Undefined:
    """Jinja's default Undefined: it prints as nothing, is false and is empty; reading into it is an error."""
    __slots__ = ("_name", "_obj")

    def __init__(self, name=None, obj=None):
        self._name = name
        self._obj = obj

    def _fail(self, *args, **kwargs):
        if self._name is None:
            raise UndefinedError("value is undefined")
        raise UndefinedError("%r is undefined" % (self._name,))

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _fail
    __floordiv__ = __rfloordiv__ = __mod__ = __rmod__ = __pos__ = __neg__ = __call__ = _fail
    __getitem__ = __lt__ = __le__ = __gt__ = __ge__ = __int__ = __float__ = __pow__ = __rpow__ = _fail

    def __getattr__(self, name):
        if name[:2] == "__":
            raise AttributeError(name)
        return self._fail()

    def __eq__(self, other):
        return type(self) is type(other)

    def __ne__(self, other):
        return type(self) is not type(other)

    def __hash__(self):
        return id(type(self))

    def __str__(self):
        return ""

    def __repr__(self):
        return "Undefined"

    def __len__(self):
        return 0

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False


# ------------------------------------------------------------------------------------------------ the lexer
_NEWLINES = re.compile(r"\r\n|\r|\n")
_TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<float>\d+\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)
  | (?P<int>\d+)
  | (?P<string>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op>\*\*|//|==|!=|<=|>=|[-+*/%<>=~|.,:()\[\]{}])
""", re.X | re.S)


def _unescape(raw):
    # Jinja decodes a string literal the way Python decodes a bytes literal's escapes, non-ASCII passing through
    return codecs.decode(raw.encode("ascii", "backslashreplace"), "unicode_escape")


def _tokenize_expression(src, pos, end_kind, line):
    """Tokens of one tag's inside, from `pos` to its closing delimiter at bracket depth 0.
    -> (tokens, position after the delimiter, whether the delimiter consumed a newline, strip-right).
    A token is (kind, value, line): kind in name, int, float, string, op."""
    tokens = []
    depth = 0
    n = len(src)
    while True:
        if pos >= n:
            raise TemplateSyntaxError("unexpected end of template, a tag is not closed (line %d)" % line)
        if depth == 0:
            if end_kind == "block":
                for opener, strip, plus in (("-%}", True, False), ("+%}", False, True), ("%}", False, False)):
                    if src.startswith(opener, pos):
                        return tokens, pos + len(opener), strip, plus
            else:
                if src.startswith("-}}", pos):
                    return tokens, pos + 3, True, False
                if src.startswith("}}", pos):
                    return tokens, pos + 2, False, False
        m = _TOKEN.match(src, pos)
        if not m:
            raise TemplateSyntaxError("unexpected character %r (line %d)" % (src[pos], line))
        kind = m.lastgroup
        text = m.group()
        if kind == "ws":
            line += text.count("\n")
        else:
            if kind == "op":
                if text in "([{":
                    depth += 1
                elif text in ")]}":
                    depth -= 1
                    if depth < 0:
                        raise TemplateSyntaxError("unexpected '%s' (line %d)" % (text, line))
            tokens.append((kind, text, line))
            line += text.count("\n")
        pos = m.end()


def _lex(source, trim_blocks, lstrip_blocks):
    """The template as a list of ('data', text) / ('var', tokens, line) / ('block', tokens, line)."""
    out = []
    pos = 0
    line = 1
    line_starting = True
    n = len(source)
    opener = re.compile(r"\{\{|\{%|\{#")
    while pos < n:
        m = opener.search(source, pos)
        if m is None:
            out.append(("data", source[pos:]))
            break
        text = source[pos:m.start()]
        kind = m.group()
        line += text.count("\n")
        p = m.end()
        sign = ""
        if p < n and source[p] in "-+":
            sign = source[p]
            p += 1
        if sign == "-":
            text = text.rstrip()
        elif sign != "+" and lstrip_blocks and kind != "{{":
            l_pos = text.rfind("\n") + 1
            if (l_pos > 0 or line_starting) and text[l_pos:] != "" and not text[l_pos:].strip():
                text = text[:l_pos]
        if text:
            out.append(("data", text))
        if kind == "{#":
            e = source.find("#}", p)
            if e < 0:
                raise TemplateSyntaxError("unexpected end of template, a comment is not closed (line %d)" % line)
            body = source[p:e]
            end = e + 2
            strip_right = body.endswith("-")
            plus = body.endswith("+")
            line += body.count("\n")
            consumed_nl = False
            if strip_right:
                while end < n and source[end].isspace():
                    consumed_nl = source[end] == "\n"
                    end += 1
            elif trim_blocks and not plus and end < n and source[end] == "\n":
                end += 1
                consumed_nl = True
            line_starting = consumed_nl
            pos = end
            continue
        toks, end, strip_right, plus = _tokenize_expression(source, p, "block" if kind == "{%" else "var", line)
        out.append(("block" if kind == "{%" else "var", toks, line))
        line += source.count("\n", p, end)
        consumed_nl = False
        if strip_right:
            while end < n and source[end].isspace():
                consumed_nl = source[end] == "\n"
                end += 1
        elif kind == "{%" and trim_blocks and not plus and end < n and source[end] == "\n":
            end += 1
            consumed_nl = True
        line_starting = consumed_nl
        pos = end
    return out


# ------------------------------------------------------------------------------------------------ the parser
_COMPARE_OPS = {"==", "!=", "<", ">", "<=", ">="}
_UNSUPPORTED = {"extends", "include", "import", "from", "block", "endblock", "call", "raw", "endraw", "with", "do",
                "autoescape", "trans", "pluralize"}


class _TokenStream:
    def __init__(self, tokens, line):
        self.tokens = tokens
        self.pos = 0
        self.line = line

    def peek(self, k=0):
        i = self.pos + k
        return self.tokens[i] if i < len(self.tokens) else ("eof", "", self.line)

    def next(self):
        t = self.peek()
        self.pos += 1
        return t

    def at_op(self, value):
        t = self.peek()
        return t[0] == "op" and t[1] == value

    def at_name(self, value):
        t = self.peek()
        return t[0] == "name" and t[1] == value

    def skip_op(self, value):
        if self.at_op(value):
            self.pos += 1
            return True
        return False

    def expect_op(self, value):
        t = self.next()
        if t[0] != "op" or t[1] != value:
            raise TemplateSyntaxError("expected '%s', got %s (line %d)" % (value, _describe(t), t[2]))
        return t

    def expect_name(self):
        t = self.next()
        if t[0] != "name":
            raise TemplateSyntaxError("expected a name, got %s (line %d)" % (_describe(t), t[2]))
        return t[1]

    @property
    def eof(self):
        return self.pos >= len(self.tokens)


def _describe(t):
    return "end of the tag" if t[0] == "eof" else repr(t[1])


class _ExprParser:
    """Jinja's precedence: condexpr < or < and < not < compare < + - < ~ < * / // % < ** < unary < postfix."""

    def __init__(self, env):
        self.env = env

    # ---- entry points
    def parse_expression(self, ts, with_condexpr=True):
        return self.parse_condexpr(ts) if with_condexpr else self.parse_or(ts)

    def parse_tuple(self, ts, with_condexpr=True, extra_end=()):
        """`a`, or `a, b, c` as a tuple (a trailing comma too)."""
        first_line = ts.peek()[2]
        items = []
        is_tuple = False
        while True:
            if items:
                if not ts.skip_op(","):
                    break
                is_tuple = True
            t = ts.peek()
            if ts.eof or (t[0] == "op" and t[1] in (")", "]", "}", "=")) or (t[0] == "name" and t[1] in extra_end):
                break
            items.append(self.parse_expression(ts, with_condexpr))
        if not is_tuple and len(items) == 1:
            return items[0]
        if not items and not is_tuple:
            raise TemplateSyntaxError("expected an expression, got %s (line %d)" % (_describe(ts.peek()), first_line))
        return ("tuple", items)

    def parse_target(self, ts):
        """The target of a for / set: a name, an attribute of a namespace, or a tuple of names."""
        items = []
        is_tuple = False
        while True:
            if items:
                if not ts.skip_op(","):
                    break
                is_tuple = True
            if ts.at_name("in") or ts.at_op("=") or ts.eof:
                break
            if ts.skip_op("("):
                items.append(self.parse_target(ts))
                ts.expect_op(")")
                continue
            name = ts.expect_name()
            node = ("name", name)
            while ts.at_op("."):
                ts.next()
                node = ("getattr", node, ts.expect_name())
            items.append(node)
        if not is_tuple and len(items) == 1:
            return items[0]
        return ("tuple", items)

    # ---- the ladder
    def parse_condexpr(self, ts):
        expr1 = self.parse_or(ts)
        while ts.at_name("if"):
            ts.next()
            expr2 = self.parse_or(ts)
            expr3 = None
            if ts.at_name("else"):
                ts.next()
                expr3 = self.parse_condexpr(ts)
            expr1 = ("cond", expr2, expr1, expr3)
        return expr1

    def parse_or(self, ts):
        left = self.parse_and(ts)
        while ts.at_name("or"):
            ts.next()
            left = ("or", left, self.parse_and(ts))
        return left

    def parse_and(self, ts):
        left = self.parse_not(ts)
        while ts.at_name("and"):
            ts.next()
            left = ("and", left, self.parse_not(ts))
        return left

    def parse_not(self, ts):
        if ts.at_name("not"):
            ts.next()
            return ("not", self.parse_not(ts))
        return self.parse_compare(ts)

    def parse_compare(self, ts):
        expr = self.parse_math1(ts)
        ops = []
        while True:
            t = ts.peek()
            if t[0] == "op" and t[1] in _COMPARE_OPS:
                ts.next()
                ops.append((t[1], self.parse_math1(ts)))
            elif t[0] == "name" and t[1] == "in":
                ts.next()
                ops.append(("in", self.parse_math1(ts)))
            elif t[0] == "name" and t[1] == "not" and ts.peek(1)[0] == "name" and ts.peek(1)[1] == "in":
                ts.next()
                ts.next()
                ops.append(("not in", self.parse_math1(ts)))
            else:
                break
        return ("compare", expr, ops) if ops else expr

    def parse_math1(self, ts):
        left = self.parse_concat(ts)
        while ts.at_op("+") or ts.at_op("-"):
            op = ts.next()[1]
            left = ("binop", op, left, self.parse_concat(ts))
        return left

    def parse_concat(self, ts):
        args = [self.parse_math2(ts)]
        while ts.at_op("~"):
            ts.next()
            args.append(self.parse_math2(ts))
        return args[0] if len(args) == 1 else ("concat", args)

    def parse_math2(self, ts):
        left = self.parse_pow(ts)
        while ts.peek()[0] == "op" and ts.peek()[1] in ("*", "/", "//", "%"):
            op = ts.next()[1]
            left = ("binop", op, left, self.parse_pow(ts))
        return left

    def parse_pow(self, ts):
        left = self.parse_unary(ts)
        while ts.at_op("**"):
            ts.next()
            left = ("binop", "**", left, self.parse_unary(ts))
        return left

    def parse_unary(self, ts, with_filter=True):
        if ts.at_op("-"):
            ts.next()
            node = ("unop", "-", self.parse_unary(ts, False))
        elif ts.at_op("+"):
            ts.next()
            node = ("unop", "+", self.parse_unary(ts, False))
        else:
            node = self.parse_primary(ts)
        node = self.parse_postfix(ts, node)
        if with_filter:
            node = self.parse_filters_and_tests(ts, node)
        return node

    def parse_primary(self, ts):
        t = ts.next()
        kind, value, line = t
        if kind == "name":
            low = value
            if low in ("true", "True"):
                return ("const", True)
            if low in ("false", "False"):
                return ("const", False)
            if low in ("none", "None"):
                return ("const", None)
            return ("name", value)
        if kind == "string":
            buf = [_unescape(value[1:-1])]
            while ts.peek()[0] == "string":
                buf.append(_unescape(ts.next()[1][1:-1]))
            return ("const", "".join(buf))
        if kind == "int":
            return ("const", int(value))
        if kind == "float":
            return ("const", float(value))
        if kind == "op":
            if value == "(":
                if ts.skip_op(")"):
                    return ("tuple", [])
                expr = self.parse_tuple(ts)
                ts.expect_op(")")
                return expr
            if value == "[":
                items = []
                while not ts.at_op("]"):
                    if items:
                        ts.expect_op(",")
                        if ts.at_op("]"):
                            break
                    items.append(self.parse_expression(ts))
                ts.expect_op("]")
                return ("list", items)
            if value == "{":
                pairs = []
                while not ts.at_op("}"):
                    if pairs:
                        ts.expect_op(",")
                        if ts.at_op("}"):
                            break
                    k = self.parse_expression(ts)
                    ts.expect_op(":")
                    pairs.append((k, self.parse_expression(ts)))
                ts.expect_op("}")
                return ("dict", pairs)
        raise TemplateSyntaxError("unexpected %s (line %d)" % (_describe(t), line))

    def parse_postfix(self, ts, node):
        while True:
            if ts.at_op("."):
                ts.next()
                t = ts.next()
                if t[0] == "name":
                    node = ("getattr", node, t[1])
                elif t[0] == "int":
                    node = ("getitem", node, ("const", int(t[1])))
                else:
                    raise TemplateSyntaxError("expected a name after '.', got %s (line %d)" % (_describe(t), t[2]))
            elif ts.at_op("["):
                ts.next()
                node = ("getitem", node, self.parse_subscript(ts))
                ts.expect_op("]")
            elif ts.at_op("("):
                node = self.parse_call(ts, node)
            else:
                return node

    def parse_subscript(self, ts):
        """`a[i]`, or a slice: `a[i:j]`, `a[i:j:k]`, `a[:]`, `a[::-1]`."""
        if ts.at_op(":"):
            parts = [None]
        else:
            first = self.parse_expression(ts)
            if not ts.at_op(":"):
                return first
            parts = [first]
        while ts.at_op(":") and len(parts) < 3:
            ts.next()
            if ts.at_op("]") or ts.at_op(":"):
                parts.append(None)
            else:
                parts.append(self.parse_expression(ts))
        while len(parts) < 3:
            parts.append(None)
        return ("slice", parts[0], parts[1], parts[2])

    def parse_call(self, ts, func):
        ts.expect_op("(")
        args = []
        kwargs = []
        while not ts.at_op(")"):
            if args or kwargs:
                ts.expect_op(",")
                if ts.at_op(")"):
                    break
            t = ts.peek()
            if t[0] == "name" and ts.peek(1)[0] == "op" and ts.peek(1)[1] == "=":
                ts.next()
                ts.next()
                kwargs.append((t[1], self.parse_expression(ts)))
            else:
                if kwargs:
                    raise TemplateSyntaxError("a positional argument follows a keyword argument (line %d)" % t[2])
                args.append(self.parse_expression(ts))
        ts.expect_op(")")
        return ("call", func, args, kwargs)

    def parse_filters_and_tests(self, ts, node):
        while True:
            if ts.at_op("|"):
                ts.next()
                name = ts.expect_name()
                while ts.at_op("."):          # a dotted filter name
                    ts.next()
                    name += "." + ts.expect_name()
                args, kwargs = [], []
                if ts.at_op("("):
                    call = self.parse_call(ts, None)
                    args, kwargs = call[2], call[3]
                node = ("filter", node, name, args, kwargs)
            elif ts.at_name("is"):
                ts.next()
                negated = False
                if ts.at_name("not"):
                    ts.next()
                    negated = True
                name = ts.expect_name()
                args, kwargs = [], []
                if ts.at_op("("):
                    call = self.parse_call(ts, None)
                    args, kwargs = call[2], call[3]
                else:
                    t = ts.peek()
                    # `x is divisibleby 3`: one bare argument, unless the next token ends the operand
                    if (t[0] in ("name", "int", "float", "string") and not (t[0] == "name" and t[1] in
                                                                              ("else", "if", "and", "or", "not",
                                                                               "is", "in"))) or \
                            (t[0] == "op" and t[1] in ("[", "{")):
                        args = [self.parse_unary(ts, with_filter=False)]
                node = ("test", node, name, args, kwargs, negated)
            elif ts.at_op("("):
                node = self.parse_call(ts, node)
            else:
                return node


class _Parser:
    def __init__(self, env, items):
        self.env = env
        self.items = items
        self.i = 0
        self.expr = _ExprParser(env)

    def parse(self):
        body = self.parse_body(())
        if self.i < len(self.items):
            raise TemplateSyntaxError("unexpected tag %r" % (self.items[self.i][1][0][1],))
        return body

    def parse_body(self, stop):
        """Statements up to a block tag whose first name is in `stop` (not consumed) or the end."""
        body = []
        while self.i < len(self.items):
            item = self.items[self.i]
            if item[0] == "data":
                body.append(("data", item[1]))
                self.i += 1
            elif item[0] == "var":
                ts = _TokenStream(item[1], item[2])
                if ts.eof:
                    raise TemplateSyntaxError("empty expression (line %d)" % item[2])
                expr = self.expr.parse_tuple(ts)
                if not ts.eof:
                    raise TemplateSyntaxError("unexpected %s (line %d)" % (_describe(ts.peek()), item[2]))
                body.append(("output", expr))
                self.i += 1
            else:
                toks = item[1]
                if not toks or toks[0][0] != "name":
                    raise TemplateSyntaxError("empty or malformed statement (line %d)" % item[2])
                name = toks[0][1]
                if name in stop:
                    return body
                self.i += 1
                ts = _TokenStream(toks[1:], item[2])
                body.append(self.parse_statement(name, ts, item[2]))
        if stop:
            raise TemplateSyntaxError("unexpected end of template: expected '%s'" % "' or '".join(sorted(stop)))
        return body

    def parse_statement(self, name, ts, line):
        if name in _UNSUPPORTED:
            raise TemplateSyntaxError("the statement '%s' is not supported here (line %d)" % (name, line))
        if name == "if":
            return self.parse_if(ts, line)
        if name == "for":
            return self.parse_for(ts, line)
        if name == "set":
            return self.parse_set(ts, line)
        if name == "macro":
            return self.parse_macro(ts, line)
        if name == "filter":
            chain = self.expr.parse_filters_and_tests(_TokenStream([("op", "|", line)] + ts.tokens[ts.pos:], line),
                                                      ("name", "\0filter"))
            body = self.parse_body({"endfilter"})
            self.end_tag({"endfilter"})
            return ("filterblock", chain, body)
        if name == "generation":
            # transformers' extension: marks the assistant's own tokens; without a mask it renders its body unchanged
            self.finish(ts, line)
            body = self.parse_body({"endgeneration"})
            self.end_tag({"endgeneration"})
            return ("group", body)
        if name in ("break", "continue"):
            self.finish(ts, line)
            return (name,)
        raise TemplateSyntaxError("unknown statement '%s' (line %d)" % (name, line))

    def finish(self, ts, line):
        if not ts.eof:
            raise TemplateSyntaxError("unexpected %s (line %d)" % (_describe(ts.peek()), line))

    def end_tag(self, expected):
        """Consumes the `{% end... %}` block tag; `expected` is the set of names allowed."""
        item = self.items[self.i]
        self.i += 1
        toks = item[1]
        if len(toks) != 1 or toks[0][1] not in expected:
            raise TemplateSyntaxError("expected '%s' (line %d)" % ("' or '".join(sorted(expected)), item[2]))

    def parse_if(self, ts, line):
        branches = []
        cond = self.expr.parse_tuple(ts, with_condexpr=False)
        self.finish(ts, line)
        body = self.parse_body({"elif", "else", "endif"})
        branches.append((cond, body))
        else_body = None
        while True:
            item = self.items[self.i]
            name = item[1][0][1]
            if name == "elif":
                self.i += 1
                ets = _TokenStream(item[1][1:], item[2])
                cond = self.expr.parse_tuple(ets, with_condexpr=False)
                self.finish(ets, item[2])
                branches.append((cond, self.parse_body({"elif", "else", "endif"})))
            elif name == "else":
                self.i += 1
                if len(item[1]) != 1:
                    raise TemplateSyntaxError("unexpected tokens after 'else' (line %d)" % item[2])
                else_body = self.parse_body({"endif"})
            else:
                self.end_tag({"endif"})
                return ("if", branches, else_body)

    def parse_for(self, ts, line):
        target = self.expr.parse_target(ts)
        if not ts.at_name("in"):
            raise TemplateSyntaxError("expected 'in' in the for statement (line %d)" % line)
        ts.next()
        iterable = self.expr.parse_tuple(ts, with_condexpr=False, extra_end=("if", "recursive"))
        cond = None
        if ts.at_name("if"):
            ts.next()
            cond = self.expr.parse_expression(ts)
        if ts.at_name("recursive"):
            raise TemplateSyntaxError("recursive loops are not supported (line %d)" % line)
        self.finish(ts, line)
        body = self.parse_body({"else", "endfor"})
        else_body = None
        item = self.items[self.i]
        if item[1][0][1] == "else":
            self.i += 1
            else_body = self.parse_body({"endfor"})
        self.end_tag({"endfor"})
        return ("for", target, iterable, cond, body, else_body)

    def parse_set(self, ts, line):
        target = self.expr.parse_target(ts)
        if ts.skip_op("="):
            value = self.expr.parse_tuple(ts)
            self.finish(ts, line)
            return ("set", target, value)
        if ts.at_op("|"):
            raise TemplateSyntaxError("a filtered set block is not supported (line %d)" % line)
        self.finish(ts, line)
        body = self.parse_body({"endset"})
        self.end_tag({"endset"})
        return ("setblock", target, body)

    def parse_macro(self, ts, line):
        name = ts.expect_name()
        ts.expect_op("(")
        params = []
        while not ts.at_op(")"):
            if params:
                ts.expect_op(",")
                if ts.at_op(")"):
                    break
            pname = ts.expect_name()
            default = None
            if ts.skip_op("="):
                default = self.expr.parse_expression(ts)
            params.append((pname, default))
        ts.expect_op(")")
        self.finish(ts, line)
        body = self.parse_body({"endmacro"})
        self.end_tag({"endmacro"})
        return ("macro", name, params, body)


# ------------------------------------------------------------------------------------------------ the sandbox
_STR_METHODS = frozenset("""capitalize casefold center count endswith expandtabs find index isalnum isalpha isascii
isdecimal isdigit isidentifier islower isnumeric isprintable isspace istitle isupper join ljust lower lstrip partition
removeprefix removesuffix replace rfind rindex rjust rpartition rsplit rstrip split splitlines startswith strip swapcase
title upper zfill""".split())
_DICT_METHODS = frozenset(["get", "items", "keys", "values", "copy"])
_LIST_METHODS = frozenset(["index", "count", "copy"])


try:
    from _string import formatter_field_name_split as _field_name_split
except ImportError:                                  # not CPython: no str.format in templates
    _field_name_split = None
import string as _string_module


class _SafeFormatter(_string_module.Formatter):
    """str.format whose `{0.attr}` and `{0[key]}` go through the same sandboxed lookups as the template's own."""

    def get_field(self, field_name, args, kwargs):
        first, rest = _field_name_split(field_name)
        obj = self.get_value(first, args, kwargs)
        for is_attr, i in rest:
            obj = _get_attr(obj, i) if is_attr else _get_item(obj, i)
        return obj, first


def _str_format(text):
    if _field_name_split is None:
        return None
    return lambda *args, **kwargs: _SafeFormatter().vformat(text, args, kwargs)


def _str_format_map(text):
    if _field_name_split is None:
        return None
    return lambda mapping: _SafeFormatter().vformat(text, (), mapping)


class Namespace:
    """`namespace(a=1)`: the one thing a template may assign to (`{% set ns.a = 2 %}`) from inside a loop."""

    def __init__(self, *args, **kwargs):
        self._attrs = dict(*args, **kwargs)

    def __repr__(self):
        return "<Namespace %r>" % (self._attrs,)


class _Loop:
    """The `loop` variable."""

    def __init__(self, items):
        self._items = items
        self._i = -1

    def attr(self, name):
        i, items = self._i, self._items
        if name == "index0":
            return i
        if name == "index":
            return i + 1
        if name == "revindex0":
            return len(items) - i - 1
        if name == "revindex":
            return len(items) - i
        if name == "first":
            return i == 0
        if name == "last":
            return i == len(items) - 1
        if name == "length":
            return len(items)
        if name == "previtem":
            return items[i - 1] if i > 0 else Undefined("there is no previous item")
        if name == "nextitem":
            return items[i + 1] if i + 1 < len(items) else Undefined("there is no next item")
        if name == "cycle":
            return lambda *args: args[i % len(args)] if args else _raise(TemplateRuntimeError("no items for cycling given"))
        if name == "changed":
            return self._changed
        return Undefined(name, self)

    _last = object()

    def _changed(self, *value):
        if self._last == value:
            return False
        self._last = value
        return True

    def __repr__(self):
        return "<loop %d/%d>" % (self._i + 1, len(self._items))


def _raise(exc):
    raise exc


class Macro:
    def __init__(self, name, params, body, frame, interp):
        self.name = name
        self.params = params
        self.body = body
        self.frame = frame
        self.interp = interp

    def __call__(self, *args, **kwargs):
        names = [p[0] for p in self.params]
        if len(args) > len(names):
            raise TemplateRuntimeError("macro %r takes not more than %d argument(s)" % (self.name, len(names)))
        for k in kwargs:
            if k not in names:
                raise TemplateRuntimeError("macro %r takes no keyword argument %r" % (self.name, k))
        frame = Frame(self.frame)
        for i, (pname, default) in enumerate(self.params):
            if i < len(args):
                frame.vars[pname] = args[i]
            elif pname in kwargs:
                frame.vars[pname] = kwargs[pname]
            elif default is not None:
                frame.vars[pname] = self.interp.eval(default, frame)
            else:
                frame.vars[pname] = Undefined(pname)
        out = []
        self.interp.run(self.body, frame, out)
        return "".join(out)


def _get_attr(obj, name):
    """`obj.name`: the attribute where a template may have one, else the item, else undefined."""
    if name[:1] == "_":
        raise SecurityError("access to attribute %r of %r object is unsafe" % (name, type(obj).__name__))
    if isinstance(obj, dict):
        if name in _DICT_METHODS:
            return getattr(obj, name)
        return obj[name] if name in obj else Undefined(name, obj)
    if isinstance(obj, str):
        if name == "format":
            method = _str_format(obj)
        elif name == "format_map":
            method = _str_format_map(obj)
        else:
            method = getattr(obj, name, None) if name in _STR_METHODS else None   # removeprefix: Python 3.9
        return method if method is not None else Undefined(name, obj)
    if isinstance(obj, (list, tuple)):
        method = getattr(obj, name, None) if name in _LIST_METHODS else None
        return method if method is not None else Undefined(name, obj)
    if isinstance(obj, Namespace):
        return obj._attrs[name] if name in obj._attrs else Undefined(name, obj)
    if isinstance(obj, _Loop):
        return obj.attr(name)
    if isinstance(obj, Macro):
        if name == "name":
            return obj.name
        if name == "arguments":
            return tuple(p[0] for p in obj.params)
        return Undefined(name, obj)
    if isinstance(obj, Undefined):
        return obj._fail()
    if isinstance(obj, (collections.abc.Mapping,)):
        return obj[name] if name in obj else Undefined(name, obj)
    return Undefined(name, obj)


def _get_item(obj, key):
    """`obj[key]`: the item, and for a string key the attribute as a fallback, else undefined."""
    if isinstance(obj, Undefined):
        return obj._fail()
    if isinstance(obj, Namespace):
        return _get_attr(obj, key) if isinstance(key, str) else Undefined(key, obj)
    if isinstance(obj, (dict, list, tuple, str)):
        try:
            return obj[key]
        except (TypeError, LookupError):
            if isinstance(key, str) and key[:1] != "_":
                return _get_attr(obj, key)
            if isinstance(key, str):
                raise SecurityError("access to attribute %r of %r object is unsafe" % (key, type(obj).__name__))
            return Undefined(key, obj)
    if isinstance(obj, collections.abc.Mapping):
        try:
            return obj[key]
        except (TypeError, LookupError):
            return Undefined(key, obj)
    if isinstance(key, str):
        return _get_attr(obj, key)
    return Undefined(key, obj)


# ------------------------------------------------------------------------------------------------ filters, tests
def _to_str(value):
    return value if isinstance(value, str) else str(value)


def _f_default(value, default_value="", boolean=False):
    if isinstance(value, Undefined) or (boolean and not value):
        return default_value
    return value


def _f_trim(value, chars=None):
    return _to_str(value).strip(chars)


def _f_join(value, d="", attribute=None):
    if attribute is not None:
        value = [_get_attr_or_item(x, attribute) for x in value]
    return _to_str(d).join(_to_str(x) for x in value)


def _get_attr_or_item(obj, attribute):
    """Jinja's make_attrgetter for `attribute='a.b'` or an index."""
    if isinstance(attribute, str):
        for part in attribute.split("."):
            obj = _get_item(obj, int(part) if part.isdigit() else part)
        return obj
    return _get_item(obj, attribute)


def _f_length(value):
    return len(value)


def _f_first(value):
    for x in value:
        return x
    return Undefined("no first item, sequence was empty")


def _f_last(value):
    for x in reversed(value):
        return x
    return Undefined("no last item, sequence was empty")


def _f_list(value):
    return list(value)


def _f_replace(s, old, new, count=None):
    s = _to_str(s)
    return s.replace(_to_str(old), _to_str(new), -1 if count is None else count)


def _f_items(value):
    if isinstance(value, Undefined):
        return
    if not isinstance(value, collections.abc.Mapping):
        raise TemplateRuntimeError("Can only get item pairs from a mapping.")
    yield from value.items()


def _f_dictsort(value, case_sensitive=False, by="key", reverse=False):
    if by == "key":
        pos = 0
    elif by == "value":
        pos = 1
    else:
        raise TemplateRuntimeError("You can only sort by either 'key' or 'value'")

    def sort_func(item):
        v = item[pos]
        if isinstance(v, str) and not case_sensitive:
            v = v.lower()
        return v

    return sorted(value.items(), key=sort_func, reverse=reverse)


def _f_sort(value, reverse=False, case_sensitive=False, attribute=None):
    def key(x):
        if attribute is not None:
            x = _get_attr_or_item(x, attribute)
        return x.lower() if isinstance(x, str) and not case_sensitive else x

    return sorted(value, key=key, reverse=reverse)


def _f_unique(value, case_sensitive=False, attribute=None):
    seen = set()
    for x in value:
        k = _get_attr_or_item(x, attribute) if attribute is not None else x
        if isinstance(k, str) and not case_sensitive:
            k = k.lower()
        if k not in seen:
            seen.add(k)
            yield x


def _f_reverse(value):
    if isinstance(value, str):
        return value[::-1]
    try:
        return reversed(value)
    except TypeError:
        rv = list(value)
        rv.reverse()
        return rv


def _f_min(value, case_sensitive=False, attribute=None):
    items = _f_sort(value, False, case_sensitive, attribute)
    return items[0] if items else Undefined("no minimum, sequence was empty")


def _f_max(value, case_sensitive=False, attribute=None):
    items = _f_sort(value, True, case_sensitive, attribute)
    return items[0] if items else Undefined("no maximum, sequence was empty")


def _f_sum(value, attribute=None, start=0):
    if attribute is not None:
        value = [_get_attr_or_item(x, attribute) for x in value]
    return sum(value, start)


def _f_int(value, default=0, base=10):
    try:
        if isinstance(value, str):
            return int(value, base)
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _f_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _f_round(value, precision=0, method="common"):
    if method == "common":
        return round(value, precision)
    import math
    func = math.ceil if method == "ceil" else math.floor
    return func(value * (10 ** precision)) / (10 ** precision)


def _f_truncate(s, length=255, killwords=False, end="...", leeway=5):
    s = _to_str(s)
    if len(s) <= length + leeway:
        return s
    if killwords:
        return s[:length - len(end)] + end
    result = s[:length - len(end)].rsplit(" ", 1)[0]
    return result + end


def _f_indent(s, width=4, first=False, blank=False):
    s = _to_str(s)
    pad = " " * width if isinstance(width, int) else _to_str(width)
    lines = s.splitlines()
    if not lines:
        return s
    out = []
    for i, line in enumerate(lines):
        if (i == 0 and not first) or (not line and not blank):
            out.append(line)
        else:
            out.append(pad + line)
    return "\n".join(out)


def _f_center(value, width=80):
    return _to_str(value).center(width)


def _f_wordcount(s):
    return len(re.findall(r"\w+", _to_str(s)))


_WORD_BEGINNING = re.compile(r"([-\s({\[<]+)")


def _f_title(s):
    return "".join(item[0].upper() + item[1:].lower() for item in _WORD_BEGINNING.split(_to_str(s)) if item)


def _f_capitalize(s):
    return _to_str(s).capitalize()


def _f_striptags(value):
    value = re.sub(r"<!--.*?-->", "", _to_str(value), flags=re.S)
    value = re.sub(r"<[^>]*>", "", value)
    return " ".join(html.unescape(value).split())


def _f_urlencode(value):
    if isinstance(value, dict) or (not isinstance(value, str) and hasattr(value, "__iter__")):
        items = value.items() if isinstance(value, dict) else value
        return "&".join("%s=%s" % (_urlquote(_to_str(k), safe="/"), _urlquote(_to_str(v), safe="/")) for k, v in items)
    return _urlquote(_to_str(value), safe="/")


def _f_format(value, *args, **kwargs):
    if args and kwargs:
        raise TemplateRuntimeError("can't handle positional and keyword arguments at the same time")
    return _to_str(value) % (kwargs or args)


def _f_batch(value, linecount, fill_with=None):
    tmp = []
    for item in value:
        if len(tmp) == linecount:
            yield tmp
            tmp = []
        tmp.append(item)
    if tmp:
        if fill_with is not None and len(tmp) < linecount:
            tmp += [fill_with] * (linecount - len(tmp))
        yield tmp


def _f_slice(value, slices, fill_with=None):
    seq = list(value)
    length = len(seq)
    items_per_slice = length // slices
    slices_with_extra = length % slices
    offset = 0
    for slice_number in range(slices):
        start = offset + slice_number * items_per_slice
        if slice_number < slices_with_extra:
            offset += 1
        end = offset + (slice_number + 1) * items_per_slice
        tmp = seq[start:end]
        if fill_with is not None and slice_number >= slices_with_extra:
            tmp.append(fill_with)
        yield tmp


def _f_groupby(value, attribute, default=None, case_sensitive=False):
    def key(x):
        v = _get_attr_or_item(x, attribute)
        if isinstance(v, Undefined) and default is not None:
            v = default
        return v.lower() if isinstance(v, str) and not case_sensitive else v

    groups = []
    for k in sorted({key(x) for x in value}, key=lambda z: (z is None, z)):
        groups.append((k, [x for x in value if key(x) == k]))
    return groups


def _f_tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    # transformers' own `tojson` for chat templates: not Jinja's HTML-safe one
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _f_abs(x):
    return abs(x)


def _f_string(x):
    return _to_str(x)


def _f_attr(obj, name):
    return _get_attr(obj, name)


def _escape_plain(x):
    """markupsafe's escape of a plain string: & < > ' "  ->  &amp; &lt; &gt; &#39; &#34;"""
    return (x.replace("&", "&amp;").replace(">", "&gt;").replace("<", "&lt;").replace("'", "&#39;")
            .replace('"', "&#34;"))


class Markup(str):
    """What `|safe` and `|e` return.  As in markupsafe, joining it with a plain string escapes the plain string:
    `"<" | safe + "<"` is `<&lt;`.  (A few templates lean on it; the rest never meet it, as output is not escaped.)"""
    __slots__ = ()

    def __add__(self, other):
        if isinstance(other, str):
            return Markup(str.__add__(self, other if isinstance(other, Markup) else _escape_plain(other)))
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, str):
            return Markup(str.__add__(other if isinstance(other, Markup) else _escape_plain(other), self))
        return NotImplemented

    def __mul__(self, n):
        return Markup(str.__mul__(self, n))

    __rmul__ = __mul__

    def __getitem__(self, key):
        return Markup(str.__getitem__(self, key))

    def join(self, seq):
        return Markup(str.join(self, [x if isinstance(x, Markup) else _escape_plain(_to_str(x)) for x in seq]))

    def replace(self, old, new, count=-1):
        esc = lambda v: v if isinstance(v, Markup) else _escape_plain(_to_str(v))
        return Markup(str.replace(self, esc(old), esc(new), count))

    def __repr__(self):
        return "Markup(%s)" % str.__repr__(self)


def _wrap_markup(name):
    method = getattr(str, name)
    setattr(Markup, name, lambda self, *a, **k: Markup(method(self, *a, **k)))


for _name in ("strip", "lstrip", "rstrip", "lower", "upper", "title", "capitalize", "swapcase", "casefold", "expandtabs"):
    _wrap_markup(_name)


def _escape(x):
    if isinstance(x, Markup):
        return x
    return Markup(_escape_plain(_to_str(x)))


def _force_escape(x):
    return Markup(_escape_plain(_to_str(x)))


def _mark_safe(x):
    return x if isinstance(x, Markup) else Markup(_to_str(x))


def _make_filter_runner(env):
    """select / reject / selectattr / rejectattr / map need to call other tests and filters by name."""

    def select_or_reject(args, kwargs, modfunc, lookup_attr):
        it = iter(args)
        value = next(it)
        attr = None
        if lookup_attr:
            attr = next(it)
        try:
            name = next(it)
            test = env.tests.get(name)
            if test is None:
                raise TemplateRuntimeError("no test named %r" % (name,))
            rest = tuple(it)
            func = lambda item: test(item, *rest)
        except StopIteration:
            func = lambda item: bool(item)
        if value:          # Jinja skips a false value (none, an empty list) instead of iterating it
            for item in value:
                probe = _get_attr_or_item(item, attr) if lookup_attr else item
                if modfunc(func(probe)):
                    yield item

    def select(value, *args, **kwargs):
        return select_or_reject((value,) + args, kwargs, lambda x: x, False)

    def reject(value, *args, **kwargs):
        return select_or_reject((value,) + args, kwargs, lambda x: not x, False)

    def selectattr(value, *args, **kwargs):
        return select_or_reject((value,) + args, kwargs, lambda x: x, True)

    def rejectattr(value, *args, **kwargs):
        return select_or_reject((value,) + args, kwargs, lambda x: not x, True)

    def map_(value, *args, **kwargs):
        if "attribute" in kwargs and not args:
            attribute = kwargs["attribute"]
            has_default = "default" in kwargs
            default = kwargs.get("default")

            def by_attribute():
                for x in value:
                    v = _get_attr_or_item(x, attribute)
                    yield default if has_default and isinstance(v, Undefined) else v
            return by_attribute()
        if args and isinstance(args[0], str):
            f = env.filters.get(args[0])
            if f is None:
                raise TemplateRuntimeError("No filter named %r." % (args[0],))
            return (f(x, *args[1:], **kwargs) for x in value)
        raise TemplateRuntimeError("map requires a filter name or attribute=")

    return {"select": select, "reject": reject, "selectattr": selectattr, "rejectattr": rejectattr, "map": map_}


def _t_divisibleby(value, num):
    return value % num == 0


def _t_in(value, seq):
    return value in seq


def _t_iterable(value):
    try:
        iter(value)
    except TypeError:
        return False
    return True


def _t_sequence(value):
    try:
        len(value)
        value.__getitem__
    except Exception:
        return False
    return True


DEFAULT_FILTERS = {
    "default": _f_default, "d": _f_default, "trim": _f_trim, "join": _f_join, "length": _f_length, "count": _f_length,
    "first": _f_first, "last": _f_last, "list": _f_list, "replace": _f_replace, "items": _f_items,
    "dictsort": _f_dictsort, "sort": _f_sort, "unique": _f_unique, "reverse": _f_reverse, "min": _f_min,
    "max": _f_max, "sum": _f_sum, "int": _f_int, "float": _f_float, "round": _f_round, "truncate": _f_truncate,
    "indent": _f_indent, "center": _f_center, "wordcount": _f_wordcount, "title": _f_title,
    "capitalize": _f_capitalize, "upper": lambda s: _to_str(s).upper(), "lower": lambda s: _to_str(s).lower(),
    "striptags": _f_striptags, "urlencode": _f_urlencode, "format": _f_format, "batch": _f_batch,
    "slice": _f_slice, "groupby": _f_groupby, "tojson": _f_tojson, "abs": _f_abs, "string": _f_string,
    "attr": _f_attr, "e": _escape, "escape": _escape, "forceescape": _force_escape, "safe": _mark_safe,
}

DEFAULT_TESTS = {
    "defined": lambda v: not isinstance(v, Undefined), "undefined": lambda v: isinstance(v, Undefined),
    "none": lambda v: v is None, "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float, complex)),
    "integer": lambda v: isinstance(v, int) and v is not True and v is not False,
    "float": lambda v: isinstance(v, float), "mapping": lambda v: isinstance(v, collections.abc.Mapping),
    "iterable": _t_iterable, "sequence": _t_sequence, "callable": callable,
    "true": lambda v: v is True, "false": lambda v: v is False, "boolean": lambda v: v is True or v is False,
    "even": lambda v: v % 2 == 0, "odd": lambda v: v % 2 == 1, "divisibleby": _t_divisibleby,
    "eq": lambda v, o: v == o, "equalto": lambda v, o: v == o, "==": lambda v, o: v == o,
    "ne": lambda v, o: v != o, "!=": lambda v, o: v != o,
    "lt": lambda v, o: v < o, "lessthan": lambda v, o: v < o, "<": lambda v, o: v < o,
    "le": lambda v, o: v <= o, "<=": lambda v, o: v <= o,
    "gt": lambda v, o: v > o, "greaterthan": lambda v, o: v > o, ">": lambda v, o: v > o,
    "ge": lambda v, o: v >= o, ">=": lambda v, o: v >= o,
    "in": _t_in, "lower": lambda v: _to_str(v).islower(), "upper": lambda v: _to_str(v).isupper(),
    "sameas": lambda v, o: v is o, "escaped": lambda v: False,
}


# ------------------------------------------------------------------------------------------------ evaluation
class Frame:
    __slots__ = ("vars", "parent")

    def __init__(self, parent=None):
        self.vars = {}
        self.parent = parent

    def lookup(self, name):
        f = self
        while f is not None:
            if name in f.vars:
                return f.vars[name]
            f = f.parent
        return _MISSING


_MISSING = object()


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class _Interpreter:
    def __init__(self, env, globals_):
        self.env = env
        self.globals = globals_

    # ---- statements
    def run(self, body, frame, out):
        for node in body:
            kind = node[0]
            if kind == "data":
                out.append(node[1])
            elif kind == "output":
                out.append(_to_str(self.eval(node[1], frame)))
            elif kind == "if":
                for cond, branch in node[1]:
                    if self.eval(cond, frame):
                        self.run(branch, frame, out)
                        break
                else:
                    if node[2] is not None:
                        self.run(node[2], frame, out)
            elif kind == "for":
                self.run_for(node, frame, out)
            elif kind == "set":
                self.assign(node[1], self.eval(node[2], frame), frame)
            elif kind == "setblock":
                buf = []
                self.run(node[2], Frame(frame), buf)
                self.assign(node[1], "".join(buf), frame)
            elif kind == "macro":
                frame.vars[node[1]] = Macro(node[1], node[2], node[3], frame, self)
            elif kind == "group":
                self.run(node[1], frame, out)
            elif kind == "filterblock":
                buf = []
                self.run(node[2], Frame(frame), buf)
                inner = Frame(frame)
                inner.vars["\0filter"] = "".join(buf)
                out.append(_to_str(self.eval(node[1], inner)))
            elif kind == "break":
                raise _Break()
            elif kind == "continue":
                raise _Continue()
            else:
                raise TemplateRuntimeError("unknown node %r" % (kind,))

    def run_for(self, node, frame, out):
        _, target, iter_node, cond, body, else_body = node
        iterable = self.eval(iter_node, frame)
        if isinstance(iterable, Undefined):
            items = []
        else:
            try:
                items = list(iterable)
            except TypeError:
                raise TemplateRuntimeError("%r is not iterable" % (type(iterable).__name__,))
        child = Frame(frame)
        if cond is not None:
            kept = []
            for item in items:
                self.assign(target, item, child)
                if self.eval(cond, child):
                    kept.append(item)
            items = kept
        if not items:
            if else_body is not None:
                self.run(else_body, frame, out)
            return
        loop = _Loop(items)
        for i, item in enumerate(items):
            loop._i = i
            child = Frame(frame)
            child.vars["loop"] = loop
            self.assign(target, item, child)
            try:
                self.run(body, child, out)
            except _Continue:
                continue
            except _Break:
                break

    def assign(self, target, value, frame):
        kind = target[0]
        if kind == "name":
            frame.vars[target[1]] = value
        elif kind == "getattr":
            obj = self.eval(target[1], frame)
            if not isinstance(obj, Namespace):
                raise TemplateRuntimeError("cannot assign attribute on %s" % type(obj).__name__)
            obj._attrs[target[2]] = value
        elif kind == "tuple":
            try:
                values = list(value)
            except TypeError:
                raise TemplateRuntimeError("cannot unpack non-iterable %s" % type(value).__name__)
            if len(values) != len(target[1]):
                raise TemplateRuntimeError("not enough values to unpack (expected %d, got %d)" % (len(target[1]), len(values)))
            for t, v in zip(target[1], values):
                self.assign(t, v, frame)
        else:
            raise TemplateRuntimeError("cannot assign to this target")

    # ---- expressions
    def eval(self, node, frame):
        kind = node[0]
        if kind == "const":
            return node[1]
        if kind == "name":
            v = frame.lookup(node[1])
            if v is _MISSING:
                v = self.globals.get(node[1], _MISSING)
                if v is _MISSING:
                    return Undefined(node[1])
            return v
        if kind == "getattr":
            return _get_attr(self.eval(node[1], frame), node[2])
        if kind == "getitem":
            obj = self.eval(node[1], frame)
            sub = node[2]
            if sub[0] == "slice":
                key = slice(*[None if p is None else self.eval(p, frame) for p in sub[1:]])
                if isinstance(obj, Undefined):
                    return obj._fail()
                try:
                    return obj[key]
                except TypeError:
                    return Undefined(key, obj)
            return _get_item(obj, self.eval(sub, frame))
        if kind == "call":
            return self.eval_call(node, frame)
        if kind == "filter":
            f = self.env.filters.get(node[2])
            if f is None:
                raise TemplateRuntimeError("No filter named %r." % (node[2],))
            value = self.eval(node[1], frame)
            args = [self.eval(a, frame) for a in node[3]]
            kwargs = {k: self.eval(v, frame) for k, v in node[4]}
            return f(value, *args, **kwargs)
        if kind == "test":
            t = self.env.tests.get(node[2])
            if t is None:
                raise TemplateRuntimeError("No test named %r." % (node[2],))
            value = self.eval(node[1], frame)
            args = [self.eval(a, frame) for a in node[3]]
            kwargs = {k: self.eval(v, frame) for k, v in node[4]}
            result = bool(t(value, *args, **kwargs))
            return (not result) if node[5] else result
        if kind == "binop":
            return self.eval_binop(node, frame)
        if kind == "unop":
            v = self.eval(node[2], frame)
            return -v if node[1] == "-" else +v
        if kind == "and":
            left = self.eval(node[1], frame)
            return self.eval(node[2], frame) if left else left
        if kind == "or":
            left = self.eval(node[1], frame)
            return left if left else self.eval(node[2], frame)
        if kind == "not":
            return not self.eval(node[1], frame)
        if kind == "compare":
            return self.eval_compare(node, frame)
        if kind == "cond":
            if self.eval(node[1], frame):
                return self.eval(node[2], frame)
            return Undefined("the condition was false and there is no else branch") if node[3] is None \
                else self.eval(node[3], frame)
        if kind == "concat":
            return "".join(_to_str(self.eval(a, frame)) for a in node[1])
        if kind == "tuple":
            return tuple(self.eval(a, frame) for a in node[1])
        if kind == "list":
            return [self.eval(a, frame) for a in node[1]]
        if kind == "dict":
            return {self.eval(k, frame): self.eval(v, frame) for k, v in node[1]}
        raise TemplateRuntimeError("unknown expression %r" % (kind,))

    def eval_call(self, node, frame):
        func = self.eval(node[1], frame)
        args = [self.eval(a, frame) for a in node[2]]
        kwargs = {k: self.eval(v, frame) for k, v in node[3]}
        if isinstance(func, Undefined):
            return func._fail()
        if not callable(func):
            raise TemplateRuntimeError("%r object is not callable" % (type(func).__name__,))
        return func(*args, **kwargs)

    def eval_binop(self, node, frame):
        op = node[1]
        a = self.eval(node[2], frame)
        b = self.eval(node[3], frame)
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b
        if op == "//":
            return a // b
        if op == "%":
            return a % b
        if op == "**":
            return a ** b
        raise TemplateRuntimeError("unknown operator %r" % (op,))

    def eval_compare(self, node, frame):
        left = self.eval(node[1], frame)
        for op, right_node in node[2]:
            right = self.eval(right_node, frame)
            if op == "==":
                ok = left == right
            elif op == "!=":
                ok = left != right
            elif op == "<":
                ok = left < right
            elif op == ">":
                ok = left > right
            elif op == "<=":
                ok = left <= right
            elif op == ">=":
                ok = left >= right
            elif op == "in":
                ok = left in right
            else:
                ok = left not in right
            if not ok:
                return False
            left = right
        return True


# ------------------------------------------------------------------------------------------------ the environment
def _safe_range(*args):
    r = range(*args)
    if len(r) > 100000:
        raise TemplateRuntimeError("range too big")
    return r


def _dict(*args, **kwargs):
    return dict(*args, **kwargs)


class Template:
    def __init__(self, env, body):
        self.env = env
        self._body = body

    def render(self, *args, **kwargs):
        context = dict(*args, **kwargs)
        frame = Frame()
        frame.vars.update(context)
        out = []
        _Interpreter(self.env, self.env.globals).run(self._body, frame, out)
        return "".join(out)


class Environment:
    """What transformers' `apply_chat_template` builds: trim_blocks and lstrip_blocks on, no trailing newline kept."""

    def __init__(self, trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False):
        self.trim_blocks = trim_blocks
        self.lstrip_blocks = lstrip_blocks
        self.keep_trailing_newline = keep_trailing_newline
        self.filters = dict(DEFAULT_FILTERS)
        self.filters.update(_make_filter_runner(self))
        self.tests = dict(DEFAULT_TESTS)
        self.globals = {"range": _safe_range, "dict": _dict, "namespace": Namespace,
                        "strftime_now": lambda fmt: datetime.datetime.now().strftime(fmt)}

    def from_string(self, source):
        source = _NEWLINES.sub("\n", source)
        if not self.keep_trailing_newline and source.endswith("\n"):
            source = source[:-1]
        items = _lex(source, self.trim_blocks, self.lstrip_blocks)
        return Template(self, _Parser(self, items).parse())
