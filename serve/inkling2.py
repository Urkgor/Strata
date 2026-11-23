"""serve/inkling2.py - Inkling 2 Mini's text for the server: its thinking, its answer and its tool calls.

Inkling 2 (the model's own chat template, serve/fixtures/inkling2/chat_template.jinja, the same bytes as the Hugging Face repo's) has no generation
prompt: a prompt ends with the last message's `<|end_message|>` and the model writes the whole reply itself, as messages:

    [<|message_model|>]<|content_thinking|>REASONING<|end_message|>
    [<|message_model|>]<|content_text|>ANSWER<|end_message|>
    <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{"k":"v"}}<|end_message|>        (one per call)
    <|content_model_end_sampling|>                                                                       (the end token: the only stop)

(the `tokenizer_config.json`'s response_template reads it the same way: reasoning_content, content and tool_calls, each repeating).  The server's
OutputParser (serve/frontend.py) reads the Qwen form: `<think>` reasoning `</think>` and `<tool_call><function=NAME><parameter=P>...`, with its streaming
of tool arguments and its recovery rules.  Inkling2Translator rewrites Inkling's text into that form as it streams (holding back only a marker that is
still arriving; a call's string arguments are streamed as they are written), and Inkling2OutputParser is the OutputParser behind it (the design of
serve/gemma4.py and serve/dsml.py).

The reasoning: the parser starts inside the reasoning when the request thinks (the default effort, 0.9, does).  The first message the model writes then
decides: `<|content_thinking|>` opens nothing (the parser is inside already) and its `<|end_message|>` becomes `</think>`; an answer or a call written
first closes the (empty) reasoning.  A thinking message of a reply that was asked not to think (effort none) is left out.
"""
from __future__ import annotations

import json
import re

from serve.frontend import OutputParser

END = "<|end_message|>"
MODEL = "<|message_model|>"
EOS = "<|content_model_end_sampling|>"
EOT = "<|endoftext|>"
_TAG_MAX = 400
# a message's head: the optional role token with the author's name (a tool call's name), then the kind of content
_HEAD = re.compile(r"(?:<\|message_model\|>([^<]*))?<\|content_(thinking|text|tool_error|invoke_tool_text|invoke_tool_json)\|>")
# the head still arriving: `<`, `<|conte`, `<|message_model|>NAME<|content_inv` ...
_HEAD_PARTIAL = re.compile(r"^(?:<(?:\|[a-z_]*\|?)?|<\|message_model\|>[^<]*(?:<(?:\|[a-z_]*\|?)?)?)$")
_CALL_OPEN = re.compile(r'\s*\{\s*"name"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"args"\s*:\s*\{')
_KEY = re.compile(r'\s*"((?:[^"\\]|\\.)*)"\s*:\s*')
_KEY_PARTIAL = re.compile(r'"(?:[^"\\]|\\.)*(?:\\|")?\s*$')
_ESC = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
_DECODER = json.JSONDecoder()
STOP_IDS = (200006,)                      # <|content_model_end_sampling|>
EFFORTS = {"none": "none", "off": "none", "disabled": "none", "false": "none", "minimal": "minimal", "low": "low", "medium": "medium", "high": "high",
           "xhigh": "xhigh", "max": "max", "maximum": "max"}


def _head(s: str):
    """The message head at the start of `s`: (match), "wait" (it may still become one) or None (it is not one)."""
    m = _HEAD.match(s)
    if m is not None:
        return m
    return "wait" if len(s) <= _TAG_MAX and _HEAD_PARTIAL.match(s) else None


def _partial_suffix(s: str, marker: str) -> int:
    """The length of the longest suffix of `s` that is a proper prefix of `marker` (what may still become it)."""
    for k in range(min(len(marker) - 1, len(s)), 0, -1):
        if s.endswith(marker[:k]):
            return k
    return 0


class Inkling2Translator:
    """Inkling 2 text -> the Qwen thinking and tool-call text, a delta at a time.  `thinking`: the parser behind starts inside the reasoning.
    on_param(tool, name, is_string) is called when a parameter's value is a string, before its text is returned, so the parser can keep a string value
    that looks like a number a string."""

    def __init__(self, thinking: bool = True, on_param=None):
        self.buf = ""
        self.state = "between"     # between | think | drop | text | call-head | args | tail
        self.think = bool(thinking)    # the parser behind is inside the reasoning
        self.tool = None
        self.phase = "key"         # key | value | str | after (inside args)
        self.key = ""
        self.esc = ""              # a JSON escape cut by the end of a delta
        self.on_param = on_param
        self.head_name = ""

    @property
    def held(self) -> str:
        """Text held back because it may be the start of a marker."""
        return self.buf

    def feed(self, delta: str) -> str:
        self.buf += delta
        return self._run(final=False)

    def flush(self) -> str:
        """The end of the output: whatever is held is text, and a call still open is closed."""
        out = self._run(final=True)
        tail: list[str] = []
        if self.state == "args":
            self._close_call(tail)
        return out + "".join(tail)

    # ------------------------------------------------------------------------------------------------
    def _run(self, final: bool) -> str:
        out: list[str] = []
        while self.buf:
            st = self.state
            if st == "between":
                if self._between(out, final) == "wait":
                    break
            elif st in ("think", "drop", "text"):
                if self._body(out, final) == "wait":
                    break
            elif st == "call-head":
                if self._call_head(out, final) == "wait":
                    break
            elif st == "args":
                if self._args(out, final) == "wait":
                    break
            else:                                               # tail: after the args' closing brace, up to the message's end
                i = self.buf.find(END)
                if i >= 0:
                    self.buf = self.buf[i + len(END):]
                    self._end_call(out)
                else:
                    k = _partial_suffix(self.buf, END)
                    self.buf = self.buf[len(self.buf) - k:] if k else ""
                    if final:
                        self.buf = ""
                        self._end_call(out)
                    break
        return "".join(out)

    def _between(self, out: list[str], final: bool):
        """Between messages: a head opens the next one; the end token and stray text are dealt with."""
        s = self.buf
        if not s.startswith("<"):
            lt = s.find("<")
            text = s if lt < 0 else s[:lt]
            if self.think and text.strip():
                out.append("</think>")                           # an answer without a head: the reasoning (if any) is over
                self.think = False
            if not self.think or text.strip():
                out.append(text)
            self.buf = s[len(text):]
            return None
        for stop in (EOS, EOT):
            if s.startswith(stop):
                self.buf = s[len(stop):]
                return None
        if s.startswith(END):                                    # a message end with no message: nothing
            self.buf = s[len(END):]
            return None
        r = _head(s)
        if r == "wait":
            if final:
                self.buf = ""
                return None
            return "wait"
        if r is None:                                            # an ordinary '<'
            out.append("<")
            self.buf = s[1:]
            return None
        self.buf = s[r.end():]
        kind, name = r.group(2), r.group(1) or ""
        if kind == "thinking":
            self.state = "think" if self.think else "drop"       # thinking asked off: a block the answer leaves out
        elif kind == "invoke_tool_json":
            if self.think:
                out.append("</think>")
                self.think = False
            self.state, self.head_name = "call-head", name
        else:
            if self.think:
                out.append("</think>")
                self.think = False
            self.state = "text"
        return None

    def _body(self, out: list[str], final: bool):
        """Inside a thinking, dropped or answer message: its text up to `<|end_message|>` (a marker's start is held back).  A message the model leaves
        without its end (the next `<|message_model|>`, or the end token, comes first) ends there as well."""
        s = self.buf
        cuts = [(s.find(t), t) for t in (END, MODEL, EOS, EOT) if t in s]
        if cuts:
            i, term = min(cuts)
            self._emit(s[:i], out)
            self.buf = s[i + len(END):] if term == END else s[i:]
            if self.state == "think":
                out.append("</think>")
                self.think = False
            self.state = "between"
            return None
        k = 0 if final else max(_partial_suffix(s, t) for t in (END, MODEL, EOS, EOT))
        self._emit(s[:len(s) - k], out)
        self.buf = s[len(s) - k:]
        return "wait" if k else None

    def _emit(self, text: str, out: list[str]) -> None:
        if text and self.state != "drop":
            out.append(text)

    # ---------------------------------------------------------------------------------------------- tool calls
    def _end_call(self, out: list[str]) -> None:
        self.state, self.phase, self.esc = "between", "key", ""

    def _close_call(self, out: list[str]) -> None:
        if self.phase == "str":
            out.append("\n</parameter>\n")
        out.append("</function>\n</tool_call>\n")
        self.state, self.phase, self.esc = "tail", "key", ""

    def _call_head(self, out: list[str], final: bool):
        """`{"name":"NAME","args":{` -> the call's opening."""
        m = _CALL_OPEN.match(self.buf)
        if m is None:
            i = self.buf.find(END)
            if i >= 0 or final or len(self.buf) > _TAG_MAX:      # not a call in this form
                raw = (self.buf[:i] if i >= 0 else self.buf).strip()
                self.buf = self.buf[i + len(END):] if i >= 0 else ""
                self.state = "between"
                name = self.head_name.strip()
                if raw.startswith("{") or not raw:                # JSON that does not parse, or nothing: the call with the text as its one argument, so that the client sees
                    if name and raw:                              # a call it can refuse (and the model try again) rather than a silence
                        out.append("<tool_call>\n<function=%s>\n<parameter=raw>\n%s\n</parameter>\n</function>\n</tool_call>\n" % (name, raw))
                elif raw:                                         # prose where a call was due (a cut or confused reply): text
                    if self.think:
                        out.append("</think>")
                        self.think = False
                    out.append(raw)
                return None
            return "wait"
        self.buf = self.buf[m.end():]
        try:
            name = json.loads('"' + m.group(1) + '"')
        except ValueError:
            name = m.group(1)
        self.tool = name or self.head_name
        self.state, self.phase = "args", "key"
        out.append("<tool_call>\n<function=%s>\n" % self.tool)
        return None

    def _args(self, out: list[str], final: bool):
        """Inside the args object: keys, strings (streamed, unescaped) and other values (read whole, written as JSON)."""
        s = self.buf
        if self.phase == "str":
            return self._string(out, final)
        s = s.lstrip()
        self.buf = s
        if not s:
            return "wait"
        if self.phase in ("key", "after"):
            if s[0] == ",":
                self.buf = s[1:]
                self.phase = "key"
                return None
            if s[0] == "}":
                self.buf = s[1:]
                self._close_call(out)
                return None
            m = _KEY.match(s)
            if m is None:
                if _KEY_PARTIAL.match(s) and not final and len(s) < _TAG_MAX:
                    return "wait"
                self._close_call(out)                            # not a key: the call ends here
                return None
            try:
                self.key = json.loads('"' + m.group(1) + '"')
            except ValueError:
                self.key = m.group(1)
            self.buf = s[m.end():]
            self.phase = "value"
            return None
        # phase == "value"
        if s[0] == '"':
            self.buf = s[1:]
            out.append("<parameter=%s>\n" % self.key)
            if self.on_param is not None:
                self.on_param(self.tool, self.key, True)
            self.phase = "str"
            return None
        try:
            value, used = _DECODER.raw_decode(s)
        except ValueError:
            if final or END in s or len(s) > 200000:
                self._close_call(out)
                return None
            return "wait"
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not final and (used >= len(s) or s[used] not in ",} \t\r\n"):
            return "wait"                                        # a number may continue in the next delta
        self.buf = s[used:]
        out.append("<parameter=%s>\n%s\n</parameter>\n" % (self.key, json.dumps(value, ensure_ascii=False)))
        self.phase = "after"
        return None

    def _string(self, out: list[str], final: bool):
        """A string value: its characters up to the closing quote, the JSON escapes read (one cut by the end of a delta waits for the rest)."""
        s = self.buf
        i, n = 0, len(s)
        piece: list[str] = []
        while i < n:
            c = s[i]
            if c == '"':
                out.append("".join(piece))
                out.append("\n</parameter>\n")
                self.buf = s[i + 1:]
                self.phase = "after"
                return None
            if c != "\\":
                piece.append(c)
                i += 1
                continue
            if i + 1 >= n:
                break                                            # the escape's letter is still to come
            d = s[i + 1]
            if d == "u":
                if i + 6 > n:
                    break
                try:
                    cp = int(s[i + 2:i + 6], 16)
                except ValueError:
                    piece.append("\\u")
                    i += 2
                    continue
                if 0xD800 <= cp < 0xDC00:                        # a surrogate pair: the low half follows as \uXXXX
                    if i + 12 > n:
                        break
                    if s[i + 6:i + 8] == "\\u":
                        try:
                            lo = int(s[i + 8:i + 12], 16)
                            if 0xDC00 <= lo < 0xE000:
                                piece.append(chr(0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)))
                                i += 12
                                continue
                        except ValueError:
                            pass
                piece.append(chr(cp))
                i += 6
                continue
            piece.append(_ESC.get(d, d))
            i += 2
        out.append("".join(piece))
        self.buf = s[i:]
        if final and self.buf:
            out.append(self.buf)
            self.buf = ""
        return "wait"


class Inkling2OutputParser(OutputParser):
    """OutputParser for Inkling 2: its messages are read as the thinking, answer and template-form calls they are."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inkling = Inkling2Translator(thinking=self.state == "reasoning", on_param=self._on_param)

    def _on_param(self, tool, name, is_string) -> None:
        """A string value ("123", "true") is a string whatever it looks like, as the model wrote it."""
        if not is_string:
            return
        spec = self.schemas.get(tool)
        if spec is None:
            return
        params = dict(spec.get("parameters") or {})
        props = dict(params.get("properties") or {})
        if (props.get(name) or {}).get("type") != "string":
            props[name] = {**(props.get(name) or {}), "type": "string"}
            params["properties"] = props
            self.schemas[tool] = {**spec, "parameters": params}
        if self.scall is not None and self.scall.name == tool:
            self.sdeclared[name] = "string"

    def feed(self, delta: str):
        return super().feed(self.inkling.feed(delta))

    def finish(self, reason: str | None = None):
        rest = self.inkling.flush()
        out = super().feed(rest) if rest else []
        out += super().finish(reason)
        return out

    @property
    def clean(self) -> bool:
        """No marker is held back anywhere: a clean point to stop the thinking at."""
        return not self.buf and not self.inkling.held


TEMPLATE_VARS: dict = {}


def to_inkling_force(force: str | None) -> str | None:
    """forced_call's opening of the call the reply must make, in the Qwen form (`<tool_call>\\n<function=NAME>\\n`) -> the same opening in Inkling's."""
    if not force:
        return force
    marker = "<function="
    at = force.find(marker)
    name = force[at + len(marker):] if at >= 0 else ""
    if name.endswith(">\n"):
        name = name[:-2]
        return MODEL + name + '<|content_invoke_tool_json|>{"name":' + json.dumps(name) + ',"args":{'
    return MODEL + name


def effort_value(value):
    """A reasoning effort as a client gives it (a word, or a number 0 .. 0.99) -> what the template takes, or ValueError."""
    if isinstance(value, bool):
        raise ValueError("unknown reasoning effort %r" % (value,))
    if isinstance(value, (int, float)):
        if not 0.0 <= float(value) <= 0.99:
            raise ValueError("reasoning effort %r: a number from 0 to 0.99" % (value,))
        return float(value)
    key = str(value).strip().lower()
    if key in EFFORTS:
        return EFFORTS[key]
    try:
        return effort_value(float(key))
    except ValueError:
        raise ValueError("unknown reasoning effort %r: use none, minimal, low, medium, high, xhigh, max or a number from 0 to 0.99" % (value,)) from None


def detect_caps(template) -> dict:
    """What this template can do, for the server's /props: read from what it renders (llama.cpp's capability names)."""
    def render(messages, tools=None, **kw):
        try:
            return template.render(messages, tools=tools, **kw)
        except Exception:                                           # noqa: BLE001 - a hint only
            return ""
    user = {"role": "user", "content": "strata_caps_user"}
    tools = [{"type": "function", "function": {"name": "strata_caps_call", "description": "strata_caps_description",
                                                "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}}}}]
    calls = [{"id": "c1", "type": "function", "function": {"name": "strata_caps_call", "arguments": {"arg": "strata_caps_arg"}}}]
    prompt = render([user], tools)
    withcalls = render([user, {"role": "assistant", "content": "", "tool_calls": calls}, {"role": "tool", "tool_call_id": "c1", "content": "strata_caps_result"}], tools)
    history = render([user, {"role": "assistant", "content": "strata_caps_answer", "reasoning_content": "strata_caps_reasoning"}, user])
    return {"supports_tools": "strata_caps_call" in prompt and "strata_caps_description" in prompt,
            "supports_tool_calls": '"name":"strata_caps_call"' in withcalls and "strata_caps_result" in withcalls,
            "supports_system_role": "strata_caps_system" in render([{"role": "system", "content": "strata_caps_system"}, user]),
            "supports_parallel_tool_calls": withcalls.count("<|content_invoke_tool_json|>") >= 1,
            "supports_preserve_reasoning": "strata_caps_reasoning" in history}
