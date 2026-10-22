"""serve/jinja_lite.py against Jinja2.

Every expectation below is Jinja2's: when Jinja2 is installed the same template is rendered with it (an
ImmutableSandboxedEnvironment with trim_blocks and lstrip_blocks, as transformers builds it) and must give the same text,
so an expectation can never be this module agreeing with itself.  The second half renders a collection of real chat
templates over many conversations with both and demands the same text, or an error from both.  Without Jinja2 the
expectations still run; the collection needs it, and the llama.cpp checkout that has it (third_party/llama.cpp-src, made
by tools/build_q35.sh, or STRATA_TEMPLATES=<a folder of .jinja files>).

    python -m unittest serve.test_jinja_lite
"""
import datetime
import json
import os
import random
import unittest
from pathlib import Path

from serve import jinja_lite as J

try:
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment
except ImportError:                                  # the server itself needs neither
    jinja2 = None

ROOT = Path(__file__).resolve().parents[1]


def lite_env():
    env = J.Environment()

    def raise_exception(message):
        raise J.TemplateError(message)

    env.globals["raise_exception"] = raise_exception
    return env


def ref_env():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
    return env


class Expect(unittest.TestCase):
    def check(self, src, expected, **ctx):
        self.assertEqual(lite_env().from_string(src).render(**ctx), expected)
        if jinja2:
            self.assertEqual(ref_env().from_string(src).render(**ctx), expected, "Jinja2 disagrees with the expectation")

    def fails(self, src, exc=Exception, **ctx):
        with self.assertRaises(exc):
            lite_env().from_string(src).render(**ctx)
        if jinja2:
            with self.assertRaises(Exception):
                ref_env().from_string(src).render(**ctx)


class Whitespace(Expect):
    def test_trim_blocks_removes_the_newline_after_a_block_not_after_a_variable(self):
        self.check("{% if true %}\nA\n{% endif %}\nB\n{{ 1 }}\nC", "A\nB\n1\nC")

    def test_lstrip_blocks_strips_the_indent_before_a_block_not_before_a_variable(self):
        self.check("a\n    {% if true %}\n    b\n    {% endif %}\n  {{ 1 }}", "a\n    b\n  1")

    def test_minus_strips_all_whitespace_on_its_side(self):
        self.check("a  \n\n  {%- if true -%}  \n\n  b  {%- endif -%}  \n\n  c", "abc")
        self.check("a {{- 1 -}} b {{ 2 -}} c", "a1b 2c")

    def test_plus_keeps_the_indent_and_the_newline(self):
        self.check("a\n   {%+ if true %}\nb{% endif %}", "a\n   b")
        self.check("{% if true +%}\nb{% endif %}", "\nb")

    def test_comments_follow_the_same_rules(self):
        self.check("a\n  {# note #}\nb {#- x -#} c", "a\nbc")

    def test_a_single_trailing_newline_goes_and_line_ends_are_normalized(self):
        self.check("x\n", "x")
        self.check("x\n\n", "x\n")
        self.check("a\r\nb\rc", "a\nb\nc")

    def test_the_start_of_the_template_counts_as_the_start_of_a_line(self):
        self.check("   {% if true %}x{% endif %}", "x")


class Scoping(Expect):
    def test_a_for_loop_has_its_own_scope_and_an_if_does_not(self):
        self.check("{% for i in [1,2] %}{% set x = i %}{% endfor %}[{{ x }}]", "[]")
        self.check("{% if true %}{% set x = 1 %}{% endif %}[{{ x }}]", "[1]")

    def test_a_namespace_is_how_a_loop_hands_a_value_out(self):
        self.check("{% set ns = namespace(n=0, last=none) %}{% for i in [3,4,5] %}{% set ns.n = ns.n + i %}"
                   "{% set ns.last = i %}{% endfor %}{{ ns.n }}/{{ ns.last }}", "12/5")

    def test_a_loop_body_starts_clean_each_time_round(self):
        self.check("{% for i in [1,2,3] %}{% if x is defined %}{{ x }}{% endif %}{% set x = i %}{% endfor %}", "")

    def test_macros_see_the_template_and_their_own_arguments(self):
        self.check("{% set g = 'hi' %}{% macro m(a, b='B') %}{{ g }} {{ a }} {{ b }}{% endmacro %}"
                   "{{ m(1) }}|{{ m(1, 2) }}|{{ m(b=3, a=4) }}", "hi 1 B|hi 1 2|hi 4 3")

    def test_a_set_block_captures_what_it_renders(self):
        self.check("{% set x %}a{{ 1 }}b{% endset %}[{{ x }}]", "[a1b]")

    def test_tuple_targets(self):
        self.check("{% set a, b = [1, 2] %}{{ a }}{{ b }}{% for k, v in {'p': 1, 'q': 2}.items() %}{{ k }}{{ v }}{% endfor %}",
                   "12p1q2")


class Loops(Expect):
    def test_loop_variables(self):
        self.check("{% for x in 'abc' %}{{ loop.index }}{{ loop.index0 }}{{ loop.revindex }}{{ loop.revindex0 }}"
                   "{{ loop.first }}{{ loop.last }}{{ loop.length }}{{ x }};{% endfor %}",
                   "1032TrueFalse3a;2121FalseFalse3b;3210FalseTrue3c;")

    def test_previtem_nextitem_cycle_changed(self):
        self.check("{% for x in [1,1,2] %}{{ loop.previtem }}|{{ loop.nextitem }}|{{ loop.cycle('a','b') }}"
                   "|{{ loop.changed(x) }};{% endfor %}", "|1|a|True;1|2|b|False;1||a|True;")

    def test_else_filter_break_continue(self):
        self.check("{% for x in [] %}a{% else %}empty{% endfor %}", "empty")
        self.check("{% for x in [1,2,3,4] if x != 2 %}{{ x }}{{ loop.length }}{% endfor %}", "133343")
        self.check("{% for x in [1,2,3,4] %}{% if x == 2 %}{% continue %}{% endif %}{% if x == 4 %}{% break %}{% endif %}"
                   "{{ x }}{% endfor %}", "13")

    def test_looping_over_a_dict_and_a_string_and_nothing(self):
        self.check("{% for k in {'a': 1, 'b': 2} %}{{ k }}{% endfor %}{% for c in 'xy' %}{{ c }}{% endfor %}"
                   "{% for z in missing %}{{ z }}{% endfor %}", "abxy")


class Expressions(Expect):
    def test_arithmetic_and_concatenation(self):
        self.check("{{ 1 + 2 * 3 }} {{ (1 + 2) * 3 }} {{ 7 // 2 }} {{ 7 % 3 }} {{ 2 ** 3 }} {{ 7 / 2 }} {{ -3 + 1 }}"
                   " {{ 'a' ~ 1 ~ 'b' }} {{ (1 + 2) ~ 3 }}", "7 9 3 1 8 3.5 -2 a1b 33")

    def test_comparisons_and_logic(self):
        self.check("{{ 1 < 2 < 3 }} {{ 1 < 2 > 3 }} {{ 'a' in 'abc' }} {{ 4 not in [1,2] }} {{ not 1 == 2 }}"
                   " {{ 0 or 'x' }} {{ 'y' and 'z' }} {{ none is none }}", "True False True True True x z True")

    def test_conditional_expressions(self):
        self.check("{{ 'a' if true else 'b' }}{{ 'a' if false else 'b' }}[{{ 'a' if false }}]", "ab[]")

    def test_literals_and_escapes(self):
        self.check(r"""{{ 'a\nb' }}|{{ "q\"q" }}|{{ 'éé' }}|{{ 'a' 'b' }}|{{ [1, 'x', [2]] }}|{{ (1,) }}|{{ {'k': 1} }}"""
                   "|{{ true }} {{ True }} {{ none }} {{ 1.5 }} {{ 2e3 }}",
                   "a\nb|q\"q|éé|ab|[1, 'x', [2]]|(1,)|{'k': 1}|True True None 1.5 2000.0")

    def test_slices_and_negative_indexes(self):
        self.check("{% set l = [1,2,3,4] %}{{ l[1:] }}{{ l[:-1] }}{{ l[::-1] }}{{ l[::2] }}{{ l[-1] }}{{ 'abc'[1:] }}",
                   "[2, 3, 4][1, 2, 3][4, 3, 2, 1][1, 3]4bc")

    def test_attribute_item_and_the_dict_methods_that_hide_a_key(self):
        d = {"name": "n", "items": "hidden", "k": {"x": [10, 20]}}
        self.check("{{ d.name }}{{ d['name'] }}{{ d.k.x[1] }}{{ d.k['x'].0 }}{{ d.nope }}|{{ d['items'] }}", "nn2010|hidden", d=d)
        self.check("{{ d.items()|list|length }}", "3", d=d)

    def test_string_methods(self):
        self.check("{{ ' a,b ' .strip().split(',') }}{{ 'abc'.startswith('a') }}{{ 'abc'.upper() }}"
                   "{{ 'a-b'.replace('-', '+') }}{{ 'x'.join(['1','2']) }}{{ 'abc'.endswith('bc') }}"
                   "{{ 'x\\ny'.splitlines() }}", "['a', 'b']TrueABCa+b1x2True['x', 'y']")


class Undefineds(Expect):
    def test_an_undefined_value_prints_nothing_and_is_false(self):
        self.check("[{{ nope }}][{{ d.nope }}]{{ 'f' if not nope }}{{ nope|length }}{{ nope|default('d') }}"
                   "{{ nope is defined }}{{ nope is not defined }}{{ d.nope is undefined }}", "[][]f0dFalseTrueTrue", d={})

    def test_reading_into_an_undefined_value_is_an_error(self):
        self.fails("{{ nope.x }}")
        self.fails("{{ d.nope.x }}", d={})
        self.fails("{{ nope[0] }}")
        self.fails("{{ nope + 1 }}")
        self.fails("{{ nope() }}")

    def test_default_and_boolean_default(self):
        self.check("{{ a|default('x') }}{{ b|default('x', true) }}{{ c|default('x') }}", "xxz", b="", c="z")


class Filters(Expect):
    def test_the_common_ones(self):
        self.check("{{ ' a '|trim }}{{ 'xxaxx'|trim('x') }}|{{ [1,2]|join(', ') }}|{{ 'abc'|length }}{{ [1]|count }}"
                   "|{{ 'AbC'|lower }}{{ 'AbC'|upper }}|{{ 'hello world'|title }}|{{ 'hello'|capitalize }}"
                   "|{{ 'a-b'|replace('-', '_') }}|{{ 'abc'|list }}|{{ [3,1,2]|sort }}|{{ [3,1,2]|first }}{{ [3,1,2]|last }}"
                   "|{{ 3|string }}{{ '5'|int + 1 }}{{ -2|abs }}{{ '2.5'|float }}|{{ [1,[2]]|tojson }}",
                   "aa|1, 2|31|abcABC|Hello World|Hello|a_b|['a', 'b', 'c']|[1, 2, 3]|32|3622.5|[1, [2]]")

    def test_tojson_is_transformers_own(self):
        self.check("{{ {'a': 'é<>'}|tojson }}|{{ [1,2]|tojson(indent=2) }}|{{ {'b': 1, 'a': 2}|tojson(sort_keys=true) }}",
                   '{"a": "é<>"}|[\n  1,\n  2\n]|{"a": 2, "b": 1}')

    def test_map_select_reject_attr_and_the_generators_they_return(self):
        items = [{"n": "a", "v": 1}, {"n": "b", "v": 0}, {"n": "c", "v": 3}]
        self.check("{{ items|map(attribute='n')|list }}{{ items|selectattr('v')|map(attribute='n')|join }}"
                   "{{ items|rejectattr('v')|map(attribute='n')|join }}{{ items|selectattr('v', 'gt', 1)|list|length }}"
                   "{{ [1,2,3,4]|select('even')|list }}{{ [1,2,3,4]|reject('odd')|list }}{{ ['a','B']|map('upper')|list }}",
                   "['a', 'b', 'c']ac" "b1[2, 4][2, 4]['A', 'B']", items=items)
        self.fails("{{ [1]|map('upper')|length }}")                     # a generator has no length, in Jinja too
        self.fails("{{ items|map(attribute='n')|tojson }}", items=items)

    def test_items_dictsort_unique_reverse_batch_sum_min_max(self):
        self.check("{{ {'b': 1, 'a': 2}|dictsort }}{{ {'a': 1}|items|list }}{{ [1,1,2]|unique|list }}"
                   "{{ [1,2]|reverse|list }}{{ 'ab'|reverse }}{{ [1,2,3]|batch(2)|list }}{{ [1,2,3]|sum }}"
                   "{{ [3,1]|min }}{{ [3,1]|max }}", "[('a', 2), ('b', 1)][('a', 1)][1, 2][2, 1]ba[[1, 2], [3]]613")

    def test_escape_and_safe_make_markup_that_escapes_what_is_added_to_it(self):
        self.check("{{ '<a href=\"x\">&\\'' |e }}|{{ '<'|safe + '<' }}|{{ '<'|e|e }}|{{ ('<'|safe) ~ '<' }}",
                   "&lt;a href=&#34;x&#34;&gt;&amp;&#39;|<&lt;|&lt;|<<")

    def test_filter_blocks_and_generation_blocks(self):
        self.check("{% filter upper %}a{{ 'b' }}{% endfilter %}|{% filter replace('x','y')|trim %}  xx {% endfilter %}",
                   "AB|yy")
        if not jinja2:
            self.assertEqual(lite_env().from_string("{% generation %}kept{% endgeneration %}").render(), "kept")

    def test_a_missing_filter_or_test_fails_only_when_it_runs(self):
        self.check("{% if false %}{{ x|nosuchfilter }}{% endif %}ok", "ok")
        self.fails("{{ 1|nosuchfilter }}")
        self.fails("{{ 1 is nosuchtest }}")

    def test_tests(self):
        self.check("{{ 1 is integer }}{{ 1.5 is float }}{{ 'a' is string }}{{ [] is iterable }}{{ {} is mapping }}"
                   "{{ true is boolean }}{{ 4 is even }}{{ 3 is odd }}{{ 9 is divisibleby 3 }}{{ 'a' is in 'abc' }}"
                   "{{ 1 is not none }}{{ 'ab' is lower }}{{ 3 is eq 3 }}{{ 3 is gt 2 }}{{ 'a' is sameas 'a' }}"
                   "{{ x is not defined }}{{ true is true }}{{ 1 is true }}{{ none is false }}",
                   "TrueTrueTrueTrueTrueTrueTrueTrueTrueTrueTrueTrueTrueTrueTrueTrueTrueFalseFalse")


class Statements(Expect):
    def test_unsupported_statements_are_refused_by_name(self):
        for src in ("{% include 'x' %}", "{% extends 'x' %}", "{% import 'x' as y %}", "{% call m() %}x{% endcall %}",
                    "{% raw %}x{% endraw %}", "{% with a = 1 %}{% endwith %}", "{% for x in y recursive %}{% endfor %}"):
            with self.assertRaises(J.TemplateSyntaxError, msg=src):
                lite_env().from_string(src)

    def test_syntax_errors(self):
        for src in ("{% if true %}", "{{ 1 +", "{% for x in %}{% endfor %}", "{{ ) }}", "{% endif %}", "{# open",
                    "{% if x %}{% else %}{% else %}{% endif %}", "{% macro m( %}{% endmacro %}"):
            with self.assertRaises(J.TemplateSyntaxError, msg=src):
                lite_env().from_string(src)

    def test_a_template_can_refuse_the_request(self):
        self.fails("{{ raise_exception('No user query found in messages.') }}")
        with self.assertRaises(J.TemplateError) as cm:
            lite_env().from_string("{{ raise_exception('nope') }}").render()
        self.assertEqual(str(cm.exception), "nope")


class Sandbox(unittest.TestCase):
    """Templates read the client's messages: they may read, and may not reach into Python or change anything."""

    def render(self, src, **ctx):
        return lite_env().from_string(src).render(**ctx)

    def test_names_that_start_with_an_underscore_are_off_limits(self):
        for src in ("{{ x.__class__ }}", "{{ x._private }}", "{{ x['__class__'] }}", "{{ ''.__class__.__mro__ }}",
                    "{{ [].__len__ }}"):
            with self.assertRaises(J.SecurityError, msg=src):
                self.render(src, x={"a": 1})

    def test_nothing_can_be_changed(self):
        self.assertEqual(self.render("[{{ l.append }}][{{ d.update }}][{{ d.pop }}][{{ l.sort }}][{{ d.clear }}]",
                                     l=[1], d={"a": 1}), "[][][][][]")
        self.assertEqual(self.render("{% set x = l.append %}{{ x is undefined }}", l=[1]), "True")

    def test_str_format_works_and_cannot_walk_attributes(self):
        self.assertEqual(self.render("{{ 'a{}b{}'.format(1, 'x') }}{{ '{0[k]}'.format(d) }}{{ '{k}'.format(k=3) }}", d={"k": 7}),
                         "a1bx73")
        with self.assertRaises(J.SecurityError):
            self.render("{{ '{0.__class__}'.format(x) }}", x=1)
        with self.assertRaises(J.SecurityError):
            self.render("{{ '{0.__class__.__init__}'.format(x) }}", x={})

    def test_only_a_namespace_takes_an_assignment_to_an_attribute(self):
        with self.assertRaises(J.TemplateError):
            self.render("{% set d.x = 1 %}", d={})

    def test_range_is_bounded_and_callables_of_the_data_are_not_reachable(self):
        with self.assertRaises(J.TemplateError):
            self.render("{{ range(10000000)|list|length }}")
        with self.assertRaises(J.TemplateError):
            self.render("{{ x() }}", x="not callable")


class Corpus(unittest.TestCase):
    """Real chat templates, many conversations: the same text as Jinja2, or an error from both."""

    TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather of a city", "parameters": {
        "type": "object", "properties": {"city": {"type": "string", "description": "the city"},
                                         "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}},
             {"type": "function", "function": {"name": "add", "description": "Adds", "parameters": {
                 "type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}}}]
    # templates that must compile and render identically: the repository's own and the Qwen family's
    MUST = ("Qwen3.5-4B.jinja", "Qwen3-Coder.jinja", "Qwen-Qwen3-0.6B.jinja", "Qwen-Qwen2.5-7B-Instruct.jinja",
            "Qwen-QwQ-32B.jinja", "meta-llama-Llama-3.1-8B-Instruct.jinja")

    @classmethod
    def conversations(cls):
        rng = random.Random(5)
        texts = ["Hello", "  padded  ", "multi\nline\n\ntext", "unicode: é ü 你好 \U0001f680", "",
                 "<think>x</think>y", "</think>\n\nanswer", "quote \" and ' and \\ end", "{{ not a tag }} {% nor this %}"]

        def msg(role, content=None, **kw):
            m = {"role": role}
            if content is not None:
                m["content"] = content
            m.update(kw)
            return m

        convs = [[msg("user", "Hi")], [msg("system", "Be brief."), msg("user", "Hi")],
                 [msg("user", "Hi"), msg("assistant", "Hello!"), msg("user", "Again")],
                 [msg("system", "S"), msg("user", "q1"), msg("assistant", "a1", reasoning_content="because"),
                  msg("user", "q2"), msg("assistant", "<think>\nr\n</think>\n\na2"), msg("user", "q3")],
                 [msg("user", [{"type": "text", "text": "look"}, {"type": "image"}, {"type": "text", "text": "!"}])],
                 [msg("user", "weather?"), msg("assistant", "", tool_calls=[{"type": "function", "function": {
                     "name": "get_weather", "arguments": {"city": "Paris", "unit": "c"}}}]), msg("tool", "sunny"),
                  msg("assistant", "It is sunny.")],
                 [msg("user", "weather?"), msg("assistant", "checking", tool_calls=[
                     {"function": {"name": "get_weather", "arguments": {"city": "Rome"}}},
                     {"function": {"name": "add", "arguments": {"a": 1, "b": 2.5}}}]), msg("tool", "a"), msg("tool", "b")],
                 [msg("developer", "dev rules"), msg("user", "x")], [msg("assistant", "starts with assistant")],
                 [msg("user", "x"), msg("user", "y")], [msg("system", "S1"), msg("system", "S2"), msg("user", "u")]]
        for _ in range(12):
            c = []
            if rng.random() < 0.5:
                c.append(msg("system", rng.choice(texts)))
            for _ in range(rng.randint(1, 4)):
                c.append(msg("user", rng.choice(texts)))
                r = rng.random()
                if r < 0.5:
                    c.append(msg("assistant", rng.choice(texts)))
                elif r < 0.7:
                    c.append(msg("assistant", rng.choice(texts), reasoning_content=rng.choice(texts)))
                elif r < 0.85:
                    c.append(msg("assistant", rng.choice(texts), tool_calls=[{"function": {"name": "add", "arguments": {
                        "a": rng.randint(0, 9), "b": rng.randint(0, 9)}}}]))
                    c.append(msg("tool", rng.choice(texts)))
            convs.append(c)
        return convs

    @classmethod
    def variants(cls):
        out = []
        for tools in (None, cls.TOOLS):
            for agp in (True, False):
                for extra in ({}, {"enable_thinking": False}, {"reasoning_effort": "low"}, {"reasoning_effort": "nonsense"}):
                    out.append(dict(tools=tools, add_generation_prompt=agp, bos_token="<s>", eos_token="</s>", **extra))
        return out

    def compare(self, path):
        src = Path(path).read_text(encoding="utf-8")
        try:
            ref = ref_env().from_string(src)
        except Exception:
            return None                                # Jinja2 cannot compile it either (transformers' own tags)
        lite = lite_env().from_string(src)             # a TemplateSyntaxError here is a failure
        same = errors = 0
        for conv in self.conversations():
            for v in self.variants():
                try:
                    a, ea = ref.render(messages=conv, **v), None
                except Exception as e:
                    a, ea = None, e
                try:
                    b, eb = lite.render(messages=conv, **v), None
                except Exception as e:
                    b, eb = None, e
                if ea or eb:
                    self.assertTrue(ea and eb, "%s: %s raised and %s did not\n%s" % (
                        Path(path).name, "Jinja2" if ea else "jinja_lite", "jinja_lite" if ea else "Jinja2",
                        json.dumps(conv)[:300]))
                    errors += 1
                else:
                    self.assertEqual(b, a, "%s differs\n%s" % (Path(path).name, json.dumps(conv)[:300]))
                    same += 1
        return same, errors

    @unittest.skipUnless(jinja2, "needs Jinja2 to compare with")
    def test_the_repositorys_own_template(self):
        same, errors = self.compare(ROOT / "serve" / "chat_template.jinja")
        self.assertGreater(same, 100)

    @unittest.skipUnless(jinja2, "needs Jinja2 to compare with")
    def test_the_collection(self):
        folder = Path(os.environ.get("STRATA_TEMPLATES") or ROOT / "third_party" / "llama.cpp-src" / "models" / "templates")
        paths = sorted(folder.glob("*.jinja")) if folder.is_dir() else []
        if not paths:
            self.skipTest("no folder of templates (set STRATA_TEMPLATES, or run tools/build_q35.sh)")
        compared = []
        for p in paths:
            r = self.compare(p)
            if r is not None:
                compared.append(p.name)
        for name in self.MUST:
            if (folder / name).exists():
                self.assertIn(name, compared)
        self.assertGreater(len(compared), len(paths) // 2)


if __name__ == "__main__":
    unittest.main()
