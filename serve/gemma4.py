"""serve/gemma4.py - Gemma 4's text for the server: its thinking channel and its tool calls.

Gemma 4 (the model's own chat template, serve/fixtures/gemma4/chat_template.jinja, the same bytes as the GGUF's) writes

    <|channel>thought\\nREASONING<channel|>ANSWER                     (the thinking, when the template asked for it)
    <|tool_call>call:get_weather{city:<|"|>Paris<|"|>,days:3}<tool_call|>      (a call: bare keys, strings between <|"|> marks, other JSON-ish values)

and then `<|tool_response>` (an end-of-generation token: the reply stops where the tool's answer goes).  The server's OutputParser
(serve/frontend.py) reads the Qwen form: `<think>` reasoning `</think>` and `<tool_call><function=NAME><parameter=P>...`, with its streaming of
tool arguments, its calls inside the thinking and its recovery rules.  Gemma4Translator rewrites Gemma's text into that form as it streams
(holding back only a marker that is still arriving), and Gemma4OutputParser is the OutputParser behind it.  So the OpenAI and Anthropic paths,
tool_choice, streaming arguments and the thinking / answer split behave for Gemma as they do for the other families, and one place knows its
spelling (the same design as serve/dsml.py).

The reasoning: the parser starts inside the reasoning when the prompt put the model there (`thinking`).  The template's thinking prompt ends with
`<|turn>model\\n` (the model writes `<|channel>thought\\n` itself: the translator drops that opening, the parser is already inside) or, after a tool's
answer, with `<|channel>thought\\n` (nothing to drop).  Either way the first `<channel|>` becomes `</think>`.  Without thinking the template writes an
empty channel into the prompt and the reply is the answer; a channel block the model writes anyway there is left out.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from serve.frontend import Event, OutputParser

CH_OPEN = "<|channel>"
CH_CLOSE = "<channel|>"
TC_OPEN = "<|tool_call>"
TC_CLOSE = "<tool_call|>"
Q = '<|"|>'
_TAG_MAX = 400
_MARKERS = (CH_OPEN, CH_CLOSE, TC_OPEN)
_CALL_HEAD = re.compile(r"<\|tool_call>call:([^{}\s<>:]+)\{")
_CALL_HEAD_PARTIAL = re.compile(r"^<\|tool_call>(?:c(?:a(?:l(?:l(?::[^{}\s<>:]*)?)?)?)?)?$")
_CHANNEL_HEAD = re.compile(r"<\|channel>([A-Za-z_]*)\n")
_KEY = re.compile(r'(?:<\|"\|>([^<]*)<\|"\|>|([A-Za-z0-9_.\-]+))\s*:')
_KEY_PARTIAL = re.compile(r'^(?:<(?:\|(?:"(?:\|(?:>[^<]*(?:<(?:\|(?:"(?:\|)?)?)?)?)?)?)?)?|[A-Za-z0-9_.\-]+\s*)$')
_SCALAR = re.compile(r"[^,}\]\s<]+")


def _holds(s: str) -> bool:
    """`s` starts with '<' and may still become one of the markers the text state reacts to."""
    return len(s) <= _TAG_MAX and (any(m.startswith(s) for m in _MARKERS) or _CALL_HEAD_PARTIAL.match(s) is not None
                                   or (s.startswith(CH_OPEN) and _CHANNEL_HEAD.match(s) is None and "\n" not in s))


def _scalar(token: str):
    if token == "true":
        return True
    if token == "false":
        return False
    if token == "null":
        return None
    try:
        return json.loads(token)
    except ValueError:
        return token


def read_value(s: str, i: int = 0, final: bool = False):
    """Gemma's argument value at s[i:] -> (python value, index after it), or None when more text is needed.  Strings are `<|"|>...<|"|>`,
    objects `{key:value,...}` (keys bare or quoted), arrays `[v,...]`; anything else up to the next `,` `}` `]` is a scalar (true, false, null,
    a number, else a bare word).  A ValueError for text that is none of these."""
    n = len(s)
    while i < n and s[i].isspace():
        i += 1
    if i >= n:
        return None
    if s.startswith(Q, i):
        j = s.find(Q, i + len(Q))
        if j < 0:
            return None
        return s[i + len(Q):j], j + len(Q)
    if Q.startswith(s[i:]):
        return None
    c = s[i]
    if c == "{":
        obj, i = {}, i + 1
        while True:
            while i < n and (s[i].isspace() or s[i] == ","):
                i += 1
            if i >= n:
                return None
            if s[i] == "}":
                return obj, i + 1
            m = _KEY.match(s, i)
            if m is None:
                if _KEY_PARTIAL.match(s[i:]):
                    return None
                raise ValueError("a key was expected at %r" % s[i:i + 20])
            r = read_value(s, m.end(), final)
            if r is None:
                return None
            obj[m.group(1) if m.group(1) is not None else m.group(2)], i = r
    if c == "[":
        arr, i = [], i + 1
        while True:
            while i < n and (s[i].isspace() or s[i] == ","):
                i += 1
            if i >= n:
                return None
            if s[i] == "]":
                return arr, i + 1
            r = read_value(s, i, final)
            if r is None:
                return None
            v, i = r
            arr.append(v)
    m = _SCALAR.match(s, i)
    if m is None:
        raise ValueError("a value was expected at %r" % s[i:i + 20])
    if m.end() >= n and not final:
        return None                                  # the token may go on in the next delta
    return _scalar(m.group(0)), m.end()


class Gemma4Translator:
    """Gemma 4 text -> the Qwen thinking and tool-call text, a delta at a time.  `thinking`: the parser behind starts inside the reasoning.
    on_param(tool, name, is_string) is called when a parameter's value is a string, before its text is returned, so the parser can keep a
    string value that looks like a number a string."""

    def __init__(self, thinking: bool = False, on_param=None):
        self.buf = ""
        self.state = "text"          # text | drop (inside a channel block the answer leaves out) | args (inside a call's braces) | tail
        self.think = bool(thinking)  # the parser behind is inside the reasoning
        self.opened = False          # the leading `<|channel>thought` of a thinking reply was read
        self.tool = None
        self.phase = "key"           # key | value | str | after  (inside args)
        self.key = ""
        self.on_param = on_param

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
        if self.state == "args":
            tail: list[str] = []
            self._close_call(tail)
            out += "".join(tail)
        return out

    # ------------------------------------------------------------------------------------------------
    def _run(self, final: bool) -> str:
        out: list[str] = []
        while self.buf:
            if self.state in ("text", "drop"):
                lt = self.buf.find("<")
                if lt < 0:
                    self._text(self.buf, out)
                    self.buf = ""
                    break
                if lt:
                    self._text(self.buf[:lt], out)
                    self.buf = self.buf[lt:]
                r = self._marker(out)
                if r == "wait":
                    if final:
                        self._text(self.buf, out)
                        self.buf = ""
                    break
                if r is None:                                   # an ordinary '<'
                    self._text("<", out)
                    self.buf = self.buf[1:]
            elif self.state == "args":
                if self._args(out, final) == "wait":
                    break
            else:                                               # tail: after the closing brace of a call
                s = self.buf.lstrip()
                if not s:
                    self.buf = ""
                    break
                if s.startswith(TC_CLOSE):
                    self.buf = s[len(TC_CLOSE):]
                    self.state = "text"
                elif TC_CLOSE.startswith(s) and not final:
                    self.buf = s
                    break
                else:
                    self.buf = s
                    self.state = "text"
        return "".join(out)

    def _text(self, text: str, out: list[str]) -> None:
        if self.state != "drop":
            out.append(text)

    def _marker(self, out: list[str]):
        """The marker at the start of self.buf: True (consumed), "wait" (it may still become one) or None (it is not one)."""
        s = self.buf
        if s.startswith(CH_OPEN):
            m = _CHANNEL_HEAD.match(s)
            if m is None:
                if "\n" not in s and len(s) <= _TAG_MAX and not s[len(CH_OPEN):].strip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"):
                    return "wait"                                # the channel's label ("thought") is still arriving
                end = len(CH_OPEN)                               # no label: the opening alone
            else:
                end = m.end()
            self.buf = s[end:]
            if self.think and not self.opened:
                self.opened = True                               # the parser is inside the reasoning already: nothing to write
            elif not self.think:
                self.state = "drop"                              # a block the answer leaves out (up to its close)
            return True
        if s.startswith(CH_CLOSE):
            self.buf = s[len(CH_CLOSE):]
            if self.think:
                out.append("</think>")
                self.think = False
            elif self.state == "drop":
                self.state = "text"
            return True
        if s.startswith(TC_OPEN):
            m = _CALL_HEAD.match(s)
            if m is None:
                return "wait" if _holds(s) else None
            self.buf = s[m.end():]
            self.tool, self.state, self.phase = m.group(1), "args", "key"
            out.append("<tool_call>\n<function=%s>\n" % m.group(1))
            return True
        if _holds(s):
            return "wait"
        return None

    def _close_call(self, out: list[str]) -> None:
        if self.phase == "str":
            out.append("\n</parameter>\n")
        out.append("</function>\n</tool_call>\n")
        self.state, self.phase = "tail", "key"

    def _args(self, out: list[str], final: bool):
        """Inside `call:NAME{ ... }`: keys, strings (streamed), other values (read whole, then written as JSON)."""
        s = self.buf
        if self.phase == "str":
            i = s.find(Q)
            if i >= 0:
                out.append(s[:i])
                self.buf = s[i + len(Q):]
                out.append("\n</parameter>\n")
                self.phase = "after"
                return None
            hold = 0                                             # a closing mark split across deltas: keep its start back
            for k in range(min(len(Q) - 1, len(s)), 0, -1):
                if s.endswith(Q[:k]):
                    hold = k
                    break
            out.append(s[:len(s) - hold])
            self.buf = s[len(s) - hold:]
            if final and self.buf:
                out.append(self.buf)
                self.buf = ""
            return "wait"
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
            if self.phase == "after":
                self.phase = "key"                               # no comma: the next key anyway
            m = _KEY.match(s)
            if m is None:
                if _KEY_PARTIAL.match(s) and not final:
                    return "wait"
                self._close_call(out)                            # not a key: the call ends here, what follows is text
                return None
            self.key = m.group(1) if m.group(1) is not None else m.group(2)
            self.buf = s[m.end():]
            self.phase = "value"
            return None
        # phase == "value"
        if s.startswith(Q):
            self.buf = s[len(Q):]
            out.append("<parameter=%s>\n" % self.key)
            if self.on_param is not None:
                self.on_param(self.tool, self.key, True)
            self.phase = "str"
            return None
        if Q.startswith(s) and not final:
            return "wait"
        try:
            r = read_value(s, 0, final)
        except ValueError:
            self._close_call(out)
            return None
        if r is None:
            if final:
                self._close_call(out)
                return None
            return "wait"
        value, used = r
        self.buf = s[used:]
        if isinstance(value, str):                               # a bare word: the text itself, and a string
            body = value
            if self.on_param is not None:
                self.on_param(self.tool, self.key, True)
        else:
            body = json.dumps(value, ensure_ascii=False)
        out.append("<parameter=%s>\n%s\n</parameter>\n" % (self.key, body))
        self.phase = "after"
        return None


class Gemma4OutputParser(OutputParser):
    """OutputParser for Gemma 4: the model's channel and calls are read as the thinking and template-form calls they are."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gemma = Gemma4Translator(thinking=self.state == "reasoning", on_param=self._on_param)

    def _on_param(self, tool, name, is_string) -> None:
        """A string value (<|"|>...<|"|>) is a string whatever it looks like ("123", "true"), as the model wrote it."""
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
        return super().feed(self.gemma.feed(delta))

    def finish(self, reason: str | None = None):
        rest = self.gemma.flush()
        out = super().feed(rest) if rest else []
        out += super().finish(reason)
        return out

    @property
    def clean(self) -> bool:
        """No marker is held back anywhere: a clean point to stop the thinking at."""
        return not self.buf and not self.gemma.held


# the server's end-of-generation ids for Gemma 4: <eos>, <|tool_response> (the model hands over to the tool) and <turn|>
STOP_IDS = (1, 50, 106)
TEMPLATE_VARS = {"bos_token": "<bos>", "eos_token": "<eos>"}
FIXTURE_TEMPLATE = Path(__file__).resolve().parent / "fixtures" / "gemma4" / "chat_template.jinja"


def to_gemma_force(force: str | None) -> str | None:
    """forced_call's opening of the call the reply must make, in the Qwen form (`<tool_call>\\n<function=NAME>\\n`) -> the same opening in Gemma's."""
    if not force:
        return force
    marker = "<function="
    at = force.find(marker)
    name = force[at + len(marker):] if at >= 0 else ""
    if name.endswith(">\n"):
        return TC_OPEN + "call:" + name[:-2] + "{"
    return TC_OPEN + "call:" + name


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
    history = render([user, {"role": "assistant", "content": "strata_caps_answer", "reasoning_content": "strata_caps_reasoning"}, user], enable_thinking=True)
    return {"supports_tools": "strata_caps_call" in prompt and "strata_caps_description" in prompt,
            "supports_tool_calls": "call:strata_caps_call{" in withcalls and "strata_caps_result" in withcalls,
            "supports_system_role": "strata_caps_system" in render([{"role": "system", "content": "strata_caps_system"}, user]),
            "supports_parallel_tool_calls": withcalls.count("<|tool_call>") >= 1,
            "supports_preserve_reasoning": "strata_caps_reasoning" in history}
