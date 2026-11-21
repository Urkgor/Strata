"""Inkling 2 Mini through the server: its chat template (the Hugging Face repo's own, rendered by serve/jinja_lite.py), the thinking / answer /
tool-call messages (streamed into the Qwen form the OutputParser reads), the reasoning effort and the forced call.

    python -m unittest serve.test_inkling2 -v
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST  # noqa: E402
from serve import frontend  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (FAMILY_MODEL_NAMES, MockEngine, Service, arch_from_tokenizer, default_model_name, model_card_sampling, serve)  # noqa: E402
from serve.inkling2 import (Inkling2OutputParser, Inkling2Translator, detect_caps, effort_value, to_inkling_force)  # noqa: E402

TEMPLATE = ROOT / "serve" / "fixtures" / "inkling2" / "chat_template.jinja"
WEATHER = {"name": "get_weather", "description": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}
CALL = '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Paris","days":3}}<|end_message|>'
THINK = "<|message_model|><|content_thinking|>I need the weather.<|end_message|>"


def translate(text, thinking=True, step=None):
    tr = Inkling2Translator(thinking)
    out = ""
    if step is None:
        out = tr.feed(text)
    else:
        for i in range(0, len(text), step):
            out += tr.feed(text[i:i + step])
    return out + tr.flush()


class Translator(unittest.TestCase):
    def test_a_call_becomes_the_qwen_form(self):
        want = ("</think><tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n"
                "</function>\n</tool_call>\n")
        self.assertEqual(translate(CALL), want)
        self.assertEqual(translate(CALL, False), want[len("</think>"):])

    def test_thinking_then_answer_then_call(self):
        got = translate(THINK + "<|message_model|><|content_text|>Hello<|end_message|>" + CALL + "<|content_model_end_sampling|>")
        self.assertTrue(got.startswith("I need the weather.</think>Hello<tool_call>"), got)
        self.assertNotIn("<|", got)

    def test_the_role_token_is_optional(self):
        self.assertEqual(translate("<|content_thinking|>hm<|end_message|><|content_text|>yes<|end_message|>"), "hm</think>yes")

    def test_thinking_off_leaves_the_thinking_out(self):
        self.assertEqual(translate(THINK + "<|message_model|><|content_text|>A<|end_message|>", False), "A")

    def test_the_split_of_the_stream_changes_nothing(self):
        text = (THINK + "<|message_model|><|content_text|>x < y <|not a marker and <b>bold</b><|end_message|>"
                + '<|message_model|>f<|content_invoke_tool_json|>{"name":"f","args":{"a":"q\\"uo\\\\te\\n\\u00e9\\ud83d\\ude00 <|x","b":[1,2],"c":{"d":"e"},"n":-12.5e1,"t":true,"z":null}}<|end_message|>'
                + CALL + "<|content_model_end_sampling|>")
        for thinking in (True, False):
            whole = translate(text, thinking)
            for step in (1, 2, 3, 5, 8, 13, 50):
                self.assertEqual(translate(text, thinking, step), whole, (thinking, step))
        self.assertIn('<parameter=a>\nq"uo\\te\né\U0001f600 <|x\n</parameter>', translate(text))
        self.assertIn("<parameter=c>\n{\"d\": \"e\"}\n</parameter>", translate(text))
        self.assertIn("<parameter=n>\n-125.0\n</parameter>", translate(text))

    def test_a_cut_call_is_closed(self):
        got = translate('<|message_model|>f<|content_invoke_tool_json|>{"name":"f","args":{"a":"par', False)
        self.assertTrue(got.endswith("par\n</parameter>\n</function>\n</tool_call>\n"), got)

    def test_a_call_that_is_not_json_is_left_out(self):
        self.assertEqual(translate("<|message_model|>f<|content_invoke_tool_json|>nonsense<|end_message|><|message_model|><|content_text|>ok<|end_message|>", False), "ok")


class Parser(unittest.TestCase):
    TOOLS = [WEATHER, {"name": "lookup", "description": "", "parameters": {"type": "object", "properties": {"id": {}}}}]

    def events(self, text, thinking=True, step=7):
        p = Inkling2OutputParser(thinking=thinking, tools=self.TOOLS, stream_tools=True)
        evs = []
        for i in range(0, len(text), step):
            evs += p.feed(text[i:i + step])
        evs += p.finish("stop")
        return evs

    def test_reasoning_answer_and_call(self):
        for step in (1, 4, 1000):
            evs = self.events(THINK + "<|message_model|><|content_text|>Sure.<|end_message|>" + CALL, True, step)
            self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), "I need the weather.")
            self.assertEqual("".join(e.text for e in evs if e.kind == "content").strip(), "Sure.")
            calls = [e.call for e in evs if e.kind == "tool_call"]
            self.assertEqual([(c.name, c.arguments) for c in calls], [("get_weather", {"city": "Paris", "days": 3})])

    def test_a_string_value_stays_a_string(self):
        evs = self.events('<|message_model|>lookup<|content_invoke_tool_json|>{"name":"lookup","args":{"id":"007"}}<|end_message|>', False)
        self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"id": "007"}])
        evs = self.events('<|message_model|>lookup<|content_invoke_tool_json|>{"name":"lookup","args":{"id":7}}<|end_message|>', False)
        self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"id": 7}])

    def test_clean_when_nothing_is_held(self):
        p = Inkling2OutputParser(thinking=False, tools=self.TOOLS)
        p.feed("<|message_model|><|content_text|>Hello <|end_mes")
        self.assertFalse(p.clean)
        p.feed("sage|>")
        self.assertTrue(p.clean)


class Template(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ct = ChatTemplate(TEMPLATE)
        cls.ct.wrap_tools = True

    def test_the_effort_comes_first_unless_a_system_message_does(self):
        u = {"role": "user", "content": "hi"}
        self.assertEqual(self.ct.render([u]), "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>hi<|end_message|>")
        self.assertIn("level: 0.7<", self.ct.render([u], reasoning_effort="medium"))
        self.assertIn("level: 0<", self.ct.render([u], reasoning_effort="none"))
        self.assertTrue(self.ct.render([{"role": "system", "content": "S"}, u]).startswith("<|message_system|><|content_text|>S<|end_message|><|message_system|>"))

    def test_no_generation_prompt(self):
        u = {"role": "user", "content": "hi"}
        self.assertEqual(self.ct.render([u], add_generation_prompt=True), self.ct.render([u], add_generation_prompt=False))

    def test_tools_calls_and_results(self):
        calls = [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"days": 3, "city": "Paris"}}}]
        text = self.ct.render([{"role": "user", "content": "w"}, {"role": "assistant", "content": "", "reasoning_content": "hm", "tool_calls": calls},
                               {"role": "tool", "tool_call_id": "c1", "content": "sunny"}], tools=[WEATHER])
        self.assertTrue(text.startswith("<|message_system|>tool_declare<|content_xml|>[{"), text)
        self.assertIn('<|message_model|><|content_thinking|>hm<|end_message|>' + CALL.replace('"city":"Paris","days":3', '"city":"Paris","days":3'), text)
        self.assertTrue(text.endswith("<|content_model_end_sampling|><|message_tool|>get_weather<|content_text|>sunny<|end_message|>"), text)

    def test_what_the_template_renders_is_what_the_parser_reads(self):
        """The template's own spelling of an assistant turn is read back as the reasoning, the answer and the call it was written from."""
        calls = [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"days": 3, "city": "Paris"}}}]
        text = self.ct.render([{"role": "user", "content": "w"}, {"role": "assistant", "content": "ok", "reasoning_content": "hm", "tool_calls": calls}], tools=[WEATHER])
        reply = text[text.index("<|message_model|>"):].replace("<|content_model_end_sampling|>", "")
        p = Inkling2OutputParser(thinking=True, tools=[WEATHER])
        evs = p.feed(reply) + p.finish("stop")
        self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), "hm")
        self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "ok")
        self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"city": "Paris", "days": 3}])

    def test_capabilities(self):
        caps = detect_caps(self.ct)
        self.assertTrue(all(caps[k] for k in ("supports_tools", "supports_tool_calls", "supports_system_role", "supports_preserve_reasoning")), caps)


class Misc(unittest.TestCase):
    def test_effort_values(self):
        self.assertEqual(effort_value("High"), "high")
        self.assertEqual(effort_value("off"), "none")
        self.assertEqual(effort_value(0.5), 0.5)
        self.assertEqual(effort_value("0.3"), 0.3)
        for bad in ("loud", 1, 1.5, True, -0.1):
            with self.assertRaises(ValueError):
                effort_value(bad)

    def test_forced_call(self):
        self.assertEqual(to_inkling_force("<tool_call>\n<function=get_weather>\n"),
                         '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{')
        self.assertEqual(to_inkling_force("<tool_call>\n<function=get_we"), "<|message_model|>get_we")
        self.assertIsNone(to_inkling_force(None))


SPECIALS = ["<|endoftext|>", "<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>", "<|content_text|>", "<|content_image|>",
            "<|content_model_end_sampling|>", "<|content_thinking|>", "<|end_message|>", "<|content_xml|>", "<|content_invoke_tool_json|>",
            "<|content_tool_error|>", "<|content_invoke_tool_text|>"]


def mini_tokenizer():
    """The real byte-level Tokenizer class over a tiny vocabulary: the 256 bytes and Inkling's special tokens, no merges."""
    toks = [ST.BYTE_TO_UNICODE[b] for b in range(256)] + SPECIALS
    types = [1] * 256 + [3] * len(SPECIALS)
    eos = toks.index("<|content_model_end_sampling|>")
    return ST.Tokenizer(toks, [], types, "inkling2", {"tokenizer.ggml.eos_token_id": eos}), eos


class InklingEngine(MockEngine):
    def __init__(self, tok, script, end):
        super().__init__(tok, script)
        self.script = tok.encode(script, parse_special=True) + [end]
        self.last_sampling = None

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_sampling = dict(sampling or {})
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class Api(unittest.TestCase):
    """The chat API against a fake engine that replays scripted Inkling text."""

    def start(self, script):
        tok, eos = mini_tokenizer()
        self.tok = tok
        engine = InklingEngine(tok, script, eos)
        svc = Service(engine, tok, ChatTemplate(TEMPLATE), model_name=default_model_name({"arch": "inkling2"}))
        svc.model_arch = "inkling2"
        svc.set_family("inkling2")
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.addCleanup(frontend.use_family, None)
        self.engine, self.svc = engine, svc
        self.base = "http://127.0.0.1:%d" % httpd.server_address[1]

    def chat(self, **extra):
        body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 400, **extra}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())

    def prompt(self):
        return self.tok.decode(self.engine.last_prompt)

    def test_thinking_answer_and_the_prompt(self):
        self.start(THINK + "<|message_model|><|content_text|>Hello there!<|end_message|>")
        status, r = self.chat()
        self.assertEqual(status, 200, r)
        self.assertEqual(self.prompt(), "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>hi<|end_message|>")
        msg = r["choices"][0]["message"]
        self.assertEqual(msg["content"].strip(), "Hello there!")
        self.assertEqual(msg["reasoning_content"], "I need the weather.")
        self.assertEqual(r["choices"][0]["finish_reason"], "stop")
        self.assertEqual(self.engine.last_sampling.get("temperature"), 1.0)

    def test_effort_reaches_the_template(self):
        self.start("<|message_model|><|content_text|>ok<|end_message|>")
        for effort, level in (("medium", "0.7"), ("low", "0.2"), ("none", "0")):
            status, r = self.chat(reasoning_effort=effort)
            self.assertEqual(status, 200, r)
            self.assertIn("level: %s<|end_message|>" % level, self.prompt())
            self.assertEqual(r["choices"][0]["message"]["content"].strip(), "ok")
        status, r = self.chat(reasoning_effort="loud")
        self.assertEqual(status, 400, r)

    def test_a_tool_call(self):
        self.start(THINK + CALL)
        tool = {"type": "function", "function": WEATHER}
        status, r = self.chat(tools=[tool])
        self.assertEqual(status, 200, r)
        self.assertIn("<|message_system|>tool_declare<|content_xml|>", self.prompt())
        choice = r["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        call = choice["message"]["tool_calls"][0]["function"]
        self.assertEqual((call["name"], json.loads(call["arguments"])), ("get_weather", {"city": "Paris", "days": 3}))

    def test_family_wiring(self):
        self.assertEqual(FAMILY_MODEL_NAMES["inkling2"], "inkling-2-mini")
        self.assertEqual(arch_from_tokenizer(None, "inkling2"), "inkling2")
        self.assertEqual(model_card_sampling("inkling2", True), {"temperature": 1.0})
        frontend.use_family("inkling2")
        try:
            self.assertEqual(frontend.effort_kwargs("high"), {"reasoning_effort": "high"})
            self.assertEqual(frontend.effort_kwargs(0.4), {"reasoning_effort": 0.4})
            self.assertEqual(frontend.effort_kwargs("none"), {"enable_thinking": False, "reasoning_effort": "none"})
            self.assertEqual(frontend.effort_kwargs(False), {"enable_thinking": False, "reasoning_effort": "none"})
        finally:
            frontend.use_family(None)
        self.assertEqual(frontend.effort_kwargs("high"), {"reasoning_effort": "xhigh"})


if __name__ == "__main__":
    unittest.main()
