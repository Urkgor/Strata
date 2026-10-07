"""The tokenizer's pre-tokenizer pattern on the standard library's `re` (no `regex` package).

tools/strata_tokenizer.py's QWEN35_PATTERN is written with \\p{L}, \\p{N}, \\p{M} and \\s; compile_pattern() turns them into
character classes built from tools/unicode_classes.py.  The first tests hold the translation and a set of splits that
`regex` gave once (they run anywhere); the second set compares with the `regex` module itself - every code point of
each class, tens of thousands of random texts, and whether the table is what `regex` gives today - and needs it.

    python -m unittest serve.test_tokenizer_classes
"""
import random
import sys
import unicodedata
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST  # noqa: E402

try:
    import regex
except ImportError:                                  # only the comparison needs it
    regex = None

# what the `regex` module splits these into (llama.cpp's qwen35 pre-tokenizer)
SPLITS = [
    ("Hello, world! It's 2024.", ['Hello', ',', ' world', '!', ' It', "'s", ' ', '2', '0', '2', '4', '.']),
    ("I've we'll they'd you're", ['I', "'ve", ' we', "'ll", ' they', "'d", ' you', "'re"]),
    ("x_j + y_2", ['x', '_j', ' +', ' y', '_', '2']),
    ("café été", ['caf\u00e9', ' e\u0301t\u00e9']),   # a combining mark stays in the word
    ("你好，世界！", ['你好', '，世界', '！']),
    ("def f(x):\n\treturn x  # c\n\n\n", ['def', ' f', '(x', '):\n', '\treturn', ' x', ' ', ' #', ' c', '\n\n\n']),
    ("a  b   c ", ['a', ' ', ' b', '  ', ' c', ' ']),
    ("٣٤ ² Ⅰ 5", ['٣', '٤', ' ', '²', ' ', 'Ⅰ', ' ', '5']),   # every digit is one piece
    ("line1\r\nline2\n \n", ['line', '1', '\r\n', 'line', '2', '\n \n']),
    ("end  ", ['end', '  ']),
    (" nbsp em", ['\xa0nbsp', ' em']),
    ("\u001fus x", ['\x1fus', ' x']),          # U+001F is not White_Space, though Python's own \s says it is
    ("emoji \U0001f680‍\U0001f4a5 ok", ['emoji', ' 🚀‍💥', ' ok']),
]


class Translation(unittest.TestCase):
    def test_splits(self):
        rx = ST.compile_pattern(ST.QWEN35_PATTERN)
        for text, expected in SPLITS:
            self.assertEqual(rx.findall(text), expected, repr(text))

    def test_the_classes_are_spliced_into_a_set_and_made_a_set_outside_one(self):
        t = ST.translate_pattern(r"\p{N}|[^\s\p{L}]")
        self.assertTrue(t.startswith("[\\U00000030-"))               # \p{N} outside: a set of its own
        self.assertIn("|[^\\U00000009-\\U0000000d", t)               # \s inside the set: its ranges, no brackets
        self.assertNotIn("\\p", t)
        self.assertNotIn("\\s", t)

    def test_other_escapes_and_literals_are_left_alone(self):
        self.assertEqual(ST.translate_pattern(r"\d+\.\w[a-z]\\"), r"\d+\.\w[a-z]\\")
        self.assertEqual(ST.compile_pattern(r"\.").findall("a.b"), ["."])

    def test_what_cannot_be_translated_is_refused(self):
        for bad in (r"[\S]", r"[\P{L}]", r"\p{Lu}", r"\pL", r"\p{L"):
            with self.assertRaises(ValueError, msg=bad):
                ST.translate_pattern(bad)

    def test_negated_classes_outside_a_set(self):
        rx = ST.compile_pattern(r"\S+")
        self.assertEqual(rx.findall("ab  cd e"), ["ab", "cd", "e"])
        self.assertEqual(ST.compile_pattern(r"\P{L}+").findall("ab12cd!"), ["12", "!"])

    def test_the_literal_alternation_of_special_tokens_uses_re_escape(self):
        tok = ST.Tokenizer(["a", "b", "<|x|>", "[y]"], [], [1, 1, 3, 4])
        self.assertEqual(tok.encode("a<|x|>b[y]", parse_special=True), [0, 2, 1, 3])


@unittest.skipUnless(regex, "needs the `regex` module to compare with")
class AgainstRegex(unittest.TestCase):
    def test_every_code_point_of_every_class(self):
        for pattern in (r"\p{L}", r"\p{M}", r"\p{N}", r"\s", r"\S", r"\P{L}"):
            a, b = regex.compile(pattern), ST.compile_pattern(pattern)
            bad = [cp for cp in range(0x110000) if (a.fullmatch(chr(cp)) is None) != (b.fullmatch(chr(cp)) is None)]
            self.assertEqual(bad, [], "%s differs at %s" % (pattern, [hex(c) for c in bad[:5]]))

    def test_random_text_splits_the_same(self):
        rng = random.Random(3)
        cats = ("Lu", "Ll", "Lo", "Mn", "Mc", "Nd", "Nl", "No", "Zs", "Cc", "Po", "Sm", "So", "Pd", "Cf", "Cn")
        pools = {c: [] for c in cats}
        for cp in range(0x110000):
            c = unicodedata.category(chr(cp))
            if c in pools and not 0xD800 <= cp <= 0xDFFF:
                pools[c].append(cp)
        pools["ws"] = [9, 10, 11, 12, 13, 28, 29, 30, 31, 32, 0x85, 0xa0, 0x1680, 0x2000, 0x200a, 0x2028, 0x2029, 0x202f,
                       0x205f, 0x3000]
        pools["ascii"] = list(range(32, 127)) + [10, 13, 9]
        keys = list(pools)
        a, b = regex.compile(ST.QWEN35_PATTERN), ST.compile_pattern(ST.QWEN35_PATTERN)
        for _ in range(20000):
            text = "".join(chr(rng.choice(pools[rng.choice(keys)])) for _ in range(rng.randint(0, 24)))
            self.assertEqual(b.findall(text), a.findall(text), repr(text))

    def test_the_table_is_what_regex_gives_today(self):
        import gen_unicode_classes
        self.assertEqual((ROOT / "tools" / "unicode_classes.py").read_text(encoding="utf-8"), gen_unicode_classes.render(),
                         "tools/unicode_classes.py is out of date: run tools/gen_unicode_classes.py")


if __name__ == "__main__":
    unittest.main()
