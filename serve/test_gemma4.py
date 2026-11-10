"""Gemma 4 through the server: its tokenizer wiring, its chat template (against Jinja2 where installed), the thinking channel and the tool calls
(streamed into the Qwen form the OutputParser reads), the stop ids and the defaults, over both APIs against a fake engine (no GPU, no model).

    python -m unittest serve.test_gemma4 -v
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
from serve.gemma4 import (Gemma4OutputParser, Gemma4Translator, STOP_IDS, detect_caps, read_value, to_gemma_force)  # noqa: E402
from serve.server import (FAMILY_MODEL_NAMES, MockEngine, Service, arch_from_tokenizer, default_model_name, model_card_sampling,  # noqa: E402
                          serve)

TEMPLATE = ROOT / "serve" / "fixtures" / "gemma4" / "chat_template.jinja"
EOS, TOOL_END = 106, 50
S = "▁"


def mini_tokenizer() -> ST.Gemma4Tokenizer:
    """The real Gemma4Tokenizer class over a tiny vocabulary: Gemma 4's special tokens at their real ids, the 256 byte tokens, no merges (every
    character is its bytes)."""
    toks = ["<filler%d>" % i for i in range(107)]
    types = [1] * 107
    for i, (t, ty) in {0: ("<pad>", 3), 1: ("<eos>", 1), 2: ("<bos>", 3), 3: ("<unk>", 3), 4: ("<mask>", 3), 46: ("<|tool>", 3), 47: ("<tool|>", 3),
                       48: ("<|tool_call>", 4), 49: ("<tool_call|>", 4), 50: ("<|tool_response>", 4), 51: ("<tool_response|>", 4), 52: ('<|"|>', 4),
                       98: ("<|think|>", 3), 100: ("<|channel>", 4), 101: ("<channel|>", 4), 105: ("<|turn>", 3), 106: ("<turn|>", 3)}.items():
        toks[i], types[i] = t, ty
    toks += ["<0x%02X>" % b for b in range(256)] + [S]
    types += [6] * 256 + [1]
    return ST.Gemma4Tokenizer(toks, [], types, {"tokenizer.ggml.bos_token_id": 2, "tokenizer.ggml.eos_token_id": 106})


class GemmaEngine(MockEngine):
    """Replays scripted completions as ids, then an end token (106, or 50 for a reply that ends in a call); keeps what it was asked."""

    def __init__(self, tok, script, end=EOS, max_context=8192):
        super().__init__(tok, script, max_context=max_context)
        self.scripts = [tok.encode(x, parse_special=True) + [end] for x in ([script] if isinstance(script, str) else script)]
        self.script = self.scripts[0]
        self.last_sampling = None
        self.last = {}

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_sampling = dict(sampling or {})
        got = 0
        try:
            for t in super().generate(ids, max_new, sampling, cancel, embeddings):
                got += 1
                yield t
        finally:
            self.last = {"generated": got, "prompt_tokens": len(ids), "prompt_ms": 50.0, "decode_ms": got * 10.0, "finish": "stop"}


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "weather",
                                              "parameters": {"type": "object", "properties": {
                                                  "city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}
CALL = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>,days:3}<tool_call|>'


def translate(text, thinking=False, step=None):
    tr = Gemma4Translator(thinking)
    out = ""
    if step is None:
        out = tr.feed(text)
    else:
        for i in range(0, len(text), step):
            out += tr.feed(text[i:i + step])
    return out + tr.flush()


class ReadValue(unittest.TestCase):
    def test_values(self):
        self.assertEqual(read_value('<|"|>a, b<|"|>,'), ("a, b", 14))
        self.assertEqual(read_value("12,")[0], 12)
        self.assertEqual(read_value("true}")[0], True)
        self.assertEqual(read_value("null,")[0], None)
        self.assertEqual(read_value("[1,<|\"|>x<|\"|>,[2]]}")[0], [1, "x", [2]])
        self.assertEqual(read_value('{a:1,<|"|>b c<|"|>:{d:[true]}},')[0], {"a": 1, "b c": {"d": [True]}})
        self.assertEqual(read_value("-1.5e2,")[0], -150.0)
        self.assertEqual(read_value("word}")[0], "word")

    def test_incomplete_text_asks_for_more(self):
        for s in ("", "12", '<|"|>abc', '<|"|', "[1,2", "{a:1", "{a:", "tru"):
            self.assertIsNone(read_value(s), s)
        self.assertEqual(read_value("tru", final=True)[0], "tru")


class Translator(unittest.TestCase):
    def test_a_call_becomes_the_qwen_form(self):
        want = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n"
                "</function>\n</tool_call>\n")
        self.assertEqual(translate(CALL), want)

    def test_nested_values_are_json(self):
        got = translate('<|tool_call>call:f{flags:[true,null],opts:{a:1,b:<|"|>x<|"|>},ok:false}<tool_call|>')
        self.assertIn("<parameter=flags>\n[true, null]\n</parameter>", got)
        self.assertIn('<parameter=opts>\n{"a": 1, "b": "x"}\n</parameter>', got)
        self.assertIn("<parameter=ok>\nfalse\n</parameter>", got)

    def test_the_split_of_the_stream_changes_nothing(self):
        text = ("<|channel>thought\nlet me <b>think</b><channel|>First <|tool_call>call:f{a:<|\"|>x<y<|\"|>,b:[1,2],c:{d:<|\"|>e<|\"|>}}<tool_call|>"
                " and a < b and <|tool_call>call:g{}<tool_call|> then <|channel>thought\nignored<channel|>done")
        for thinking in (True, False):
            whole = translate(text, thinking)
            for step in (1, 2, 3, 5, 8, 13):
                self.assertEqual(translate(text, thinking, step), whole, (thinking, step))

    def test_thinking_first_close_is_the_end_of_the_thinking(self):
        self.assertEqual(translate("<|channel>thought\nhm<channel|>Answer", True), "hm</think>Answer")
        self.assertEqual(translate("hm<channel|>Answer", True), "hm</think>Answer")                  # the prompt ended with the opening
        self.assertEqual(translate("<|channel>thought\n<channel|>Answer", True), "</think>Answer")

    def test_without_thinking_an_empty_channel_or_a_block_is_left_out(self):
        self.assertEqual(translate("<|channel>thought\n<channel|>Answer", False), "Answer")
        self.assertEqual(translate("A<|channel>thought\nsecret<channel|>B", False), "AB")
        self.assertEqual(translate("stray <channel|>close", False), "stray close")

    def test_ordinary_angle_brackets_pass(self):
        for s in ("a < b", "if x<3 {y}", "<div>x</div>", "<", "<|", "tail <|tool"):
            self.assertEqual(translate(s), s, s)

    def test_a_call_cut_off_is_closed(self):
        got = translate('<|tool_call>call:f{a:<|"|>unfinished')
        self.assertTrue(got.endswith("unfinished\n</parameter>\n</function>\n</tool_call>\n"), got)
        got = translate("<|tool_call>call:f{a:12")
        self.assertIn("<parameter=a>\n12\n</parameter>", got)
        self.assertTrue(got.endswith("</function>\n</tool_call>\n"))

    def test_text_after_a_call_and_two_calls(self):
        got = translate(CALL + "<|tool_call>call:h{}<tool_call|>after")
        self.assertEqual(got.count("<tool_call>"), 2)
        self.assertTrue(got.endswith("</tool_call>\nafter"), got)

    def test_a_string_with_marker_lookalikes(self):
        got = translate('<|tool_call>call:f{a:<|"|>x <|" y <|<|"|>}<tool_call|>')
        self.assertIn("<parameter=a>\nx <|\" y <|\n</parameter>", got)

    def test_force_text(self):
        self.assertEqual(to_gemma_force("<tool_call>\n<function=get_weather>\n"), "<|tool_call>call:get_weather{")
        self.assertEqual(to_gemma_force("<tool_call>\n<function=get_we"), "<|tool_call>call:get_we")
        self.assertIsNone(to_gemma_force(None))
        # the forced opening, then the model's arguments, is a whole call
        self.assertEqual(translate(to_gemma_force("<tool_call>\n<function=f>\n") + 'a:<|"|>x<|"|>}'),
                         "<tool_call>\n<function=f>\n<parameter=a>\nx\n</parameter>\n</function>\n</tool_call>\n")


class Parser(unittest.TestCase):
    TOOLS = [WEATHER["function"], {"name": "lookup", "description": "", "parameters": {"type": "object", "properties": {"id": {}}}}]

    def events(self, text, thinking=False, step=7):
        p = Gemma4OutputParser(thinking=thinking, tools=self.TOOLS, stream_tools=True)
        evs = []
        for i in range(0, len(text), step):
            evs += p.feed(text[i:i + step])
        evs += p.finish("stop")
        return evs

    def test_reasoning_then_answer(self):
        for step in (1, 4, 1000):
            evs = self.events("<|channel>thought\nThink < ok.<channel|>The answer.", True, step)
            self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), "Think < ok.")
            self.assertEqual("".join(e.text for e in evs if e.kind == "content").strip(), "The answer.")

    def test_a_call_with_arguments_streamed(self):
        for step in (1, 5, 1000):
            evs = self.events(CALL, False, step)
            calls = [e.call for e in evs if e.kind == "tool_call"]
            self.assertEqual([(c.name, c.arguments) for c in calls], [("get_weather", {"city": "Paris", "days": 3})])
            self.assertEqual("".join(e.text for e in evs if e.kind == "tool_args"), '{"city":"Paris","days":3}')

    def test_a_string_value_stays_a_string(self):
        evs = self.events('<|tool_call>call:lookup{id:<|"|>007<|"|>}<tool_call|>')
        self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"id": "007"}])
        evs = self.events("<|tool_call>call:lookup{id:7}<tool_call|>")
        self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"id": 7}])

    def test_thinking_then_a_call(self):
        evs = self.events("<|channel>thought\nI need the weather.<channel|>" + CALL, True, 3)
        self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), "I need the weather.")
        self.assertEqual([e.call.name for e in evs if e.kind == "tool_call"], ["get_weather"])

    def test_clean_when_nothing_is_held(self):
        p = Gemma4OutputParser(thinking=False, tools=self.TOOLS)
        p.feed("Hello <|tool_c")
        self.assertFalse(p.clean)
        p.feed('all:lookup{id:1}<tool_call|>')
        self.assertTrue(p.clean)


class Template(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ct = ChatTemplate(TEMPLATE)
        cls.ct.defaults.update({"bos_token": "<bos>", "eos_token": "<eos>"})

    def test_a_turn(self):
        self.assertEqual(self.ct.render([{"role": "user", "content": "Hello"}]), "<bos><|turn>user\nHello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")

    def test_thinking_on(self):
        got = self.ct.render([{"role": "user", "content": "Hello"}], enable_thinking=True)
        self.assertEqual(got, "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n")

    def test_a_call_and_its_result(self):
        msgs = [{"role": "user", "content": "W?"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}}]},
                {"role": "tool", "tool_call_id": "c", "content": "sunny"}]
        got = self.ct.render(msgs, tools=[WEATHER])
        self.assertIn("<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:3}<tool_call|>", got)
        self.assertIn("<|tool_response>response:get_weather{value:<|\"|>sunny<|\"|>}<tool_response|>", got)
        self.assertTrue(got.endswith("<tool_response|>"), got[-80:])

    def test_the_caps(self):
        caps = detect_caps(self.ct)
        self.assertEqual(caps, {"supports_tools": True, "supports_tool_calls": True, "supports_system_role": True, "supports_parallel_tool_calls": True,
                                "supports_preserve_reasoning": False})      # (the template drops the reasoning of earlier turns)

    def test_against_jinja2(self):
        try:
            import jinja2
        except ImportError:
            self.skipTest("jinja2 is not installed")
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
        ref = env.from_string(TEMPLATE.read_text(encoding="utf-8"))
        cases = [dict(messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "Again"}]),
                 dict(messages=[{"role": "user", "content": "W?"}], tools=[WEATHER], enable_thinking=True),
                 dict(messages=[{"role": "user", "content": "W?"}, {"role": "assistant", "content": "", "reasoning_content": "call it",
                                "tool_calls": [{"id": "c", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
                                {"role": "tool", "tool_call_id": "c", "content": [{"type": "text", "text": "cold"}]}], tools=[WEATHER], enable_thinking=True)]
        for kw in cases:
            kw = dict(kw)
            msgs = kw.pop("messages")
            self.assertEqual(self.ct.render(msgs, **kw), ref.render(messages=msgs, add_generation_prompt=True, bos_token="<bos>", eos_token="<eos>", **kw))


class Tokenizer(unittest.TestCase):
    def test_the_special_tokens_and_the_end_ids(self):
        tok = mini_tokenizer()
        self.assertEqual(tok.encode("<|turn>user\nhi<turn|>", parse_special=True)[0], 105)
        self.assertEqual(tok.encode("<turn|>", parse_special=True), [106])
        self.assertEqual(tok.encode("<|tool_response>", parse_special=True), [50])
        self.assertEqual(tok.decode(tok.encode("héllo\n", parse_special=True)), "héllo\n")
        self.assertEqual(STOP_IDS, (1, 50, 106))


class Wiring(unittest.TestCase):
    def test_family_tables(self):
        self.assertEqual(FAMILY_MODEL_NAMES["gemma4"], "gemma-4-26b-a4b")
        self.assertEqual(default_model_name({"arch": "gemma4"}), "gemma-4-26b-a4b")
        self.assertEqual(arch_from_tokenizer(None, "gemma4"), "gemma4")
        self.assertEqual(arch_from_tokenizer("qwen35moe", "gemma4"), "qwen35moe")
        self.assertEqual(model_card_sampling("gemma4", True), {"temperature": 1.0, "top_p": 0.95, "top_k": 64})
        self.assertEqual(model_card_sampling("gemma4", False), model_card_sampling("gemma4", True))

    def test_effort_words_switch_the_thinking(self):
        frontend.use_family("gemma4")
        try:
            self.assertEqual(frontend.effort_kwargs("high"), {"enable_thinking": True})
            self.assertEqual(frontend.effort_kwargs("low"), {"enable_thinking": True})
            self.assertEqual(frontend.effort_kwargs("none"), {"enable_thinking": False})
            self.assertEqual(frontend.effort_kwargs(False), {"enable_thinking": False})
            self.assertEqual(frontend.budget_effort(4000), {"enable_thinking": True})
        finally:
            frontend.use_family(None)
        self.assertEqual(frontend.effort_kwargs("high"), {"reasoning_effort": "xhigh"})


class ApiBase(unittest.TestCase):
    """Both APIs against a fake engine that replays scripted Gemma text."""

    @classmethod
    def setUpClass(cls):
        cls.tok = mini_tokenizer()

    def start(self, script, end=EOS):
        tok = self.tok
        engine = GemmaEngine(tok, script, end=end)
        svc = Service(engine, tok, ChatTemplate(TEMPLATE), model_name=default_model_name({"arch": "gemma4"}))
        svc.model_arch = "gemma4"
        svc.set_family("gemma4")
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.addCleanup(frontend.use_family, None)
        self.engine, self.svc = engine, svc
        self.base = f"http://127.0.0.1:{httpd.server_address[1]}"
        return engine

    def post(self, path, body, raw=False):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read().decode()
                return r.status, (data if raw else json.loads(data))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())

    def prompt(self):
        return self.tok.decode(self.engine.last_prompt)

    def chat(self, **extra):
        body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 400, **extra}
        return self.post("/v1/chat/completions", body)


class Api(ApiBase):
    def test_no_thinking_by_default(self):
        self.start("Hello there!")
        status, r = self.chat()
        self.assertEqual(status, 200, r)
        self.assertTrue(self.prompt().endswith("<|turn>user\nhi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"), self.prompt())
        msg = r["choices"][0]["message"]
        self.assertEqual(msg["content"].strip(), "Hello there!")
        self.assertFalse(msg.get("reasoning_content"))
        self.assertEqual(r["choices"][0]["finish_reason"], "stop")
        self.assertEqual(self.engine.last_sampling.get("top_k"), 64)

    def test_thinking_when_asked(self):
        for kw in ({"reasoning_effort": "high"}, {"chat_template_kwargs": {"enable_thinking": True}}):
            with self.subTest(kw=kw):
                self.start("<|channel>thought\nThe user greets me.<channel|>Hello!")
                status, r = self.chat(**kw)
                self.assertEqual(status, 200, r)
                self.assertTrue(self.prompt().startswith("<bos><|turn>system\n<|think|>\n<turn|>\n"), self.prompt())
                self.assertTrue(self.prompt().endswith("<|turn>model\n"), self.prompt())
                msg = r["choices"][0]["message"]
                self.assertEqual(msg["reasoning_content"], "The user greets me.")
                self.assertEqual(msg["content"].strip(), "Hello!")

    def test_anthropic_thinking(self):
        self.start("<|channel>thought\nHm.<channel|>Hello!")
        status, r = self.post("/v1/messages", {"model": "x", "max_tokens": 400, "messages": [{"role": "user", "content": "hi"}],
                                               "thinking": {"type": "enabled", "budget_tokens": 4000}})
        self.assertEqual(status, 200, r)
        self.assertEqual([b["type"] for b in r["content"]], ["thinking", "text"])
        self.assertEqual(r["content"][1]["text"].strip(), "Hello!")

    def test_a_tool_call(self):
        self.start(CALL, end=TOOL_END)
        status, r = self.chat(tools=[WEATHER])
        self.assertEqual(status, 200, r)
        self.assertIn("<|tool>declaration:get_weather{", self.prompt())
        choice = r["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        calls = choice["message"]["tool_calls"]
        self.assertEqual([(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls], [("get_weather", {"city": "Paris", "days": 3})])

    def test_a_tool_call_streamed(self):
        self.start(CALL, end=TOOL_END)
        status, raw = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 400, "tools": [WEATHER],
                                                         "stream": True}, raw=True)
        self.assertEqual(status, 200)
        chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        args = "".join(tc["function"].get("arguments") or "" for c in chunks for tc in (c["choices"][0]["delta"].get("tool_calls") or []))
        self.assertEqual(json.loads(args), {"city": "Paris", "days": 3})
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_the_result_goes_back_into_the_prompt(self):
        self.start("It is sunny.")
        msgs = [{"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]
        status, r = self.post("/v1/chat/completions", {"messages": msgs, "tools": [WEATHER], "max_tokens": 200})
        self.assertEqual(status, 200, r)
        self.assertTrue(self.prompt().endswith("<|tool_response>response:get_weather{value:<|\"|>sunny<|\"|>}<tool_response|>"), self.prompt()[-200:])
        self.assertEqual(r["choices"][0]["message"]["content"].strip(), "It is sunny.")

    def test_a_forced_tool(self):
        self.start('city:<|"|>Rome<|"|>,days:2}<tool_call|>', end=TOOL_END)
        status, r = self.chat(tools=[WEATHER], tool_choice={"type": "function", "function": {"name": "get_weather"}})
        self.assertEqual(status, 200, r)
        self.assertTrue(self.prompt().endswith("<|tool_call>call:get_weather{"), self.prompt()[-80:])
        calls = r["choices"][0]["message"]["tool_calls"]
        self.assertEqual([json.loads(c["function"]["arguments"]) for c in calls], [{"city": "Rome", "days": 2}])


if __name__ == "__main__":
    unittest.main()
