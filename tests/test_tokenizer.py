from src.Indexor.Chunker.Tokenizer.MarkDownTokenizer import MarkDownTokenizer
from src.Indexor.Chunker.Tokenizer.PythonTokenizer import PythonTokenizer


def test_tokenize_01():
    tokenizer = MarkDownTokenizer()
    result = tokenizer.tokenize(
        "test_variable"
    )

    assert ["test", "variable"] == result


def test_tokenize_02():
    tokenizer = MarkDownTokenizer()
    result = tokenizer.tokenize(
        "this a string that is being tested"
    )

    assert ["this", "a", "string", "that", "is", "being", "tested"] == result


def test_tokenize_03():
    tokenizer = MarkDownTokenizer()
    result = tokenizer.tokenize("")

    assert [] == result


def test_tokenize_04():
    tokenizer = MarkDownTokenizer()
    result = tokenizer.tokenize(
        "this\nis\nmulti\nline"
    )

    assert ["this", "is", "multi", "line"] == result


def test_tokenize_05():
    tokenizer = MarkDownTokenizer()
    result = tokenizer.tokenize(
        "this56 a string67 that 555 is being tested"
    )

    assert ["this56", "a", "string67", "that", "555", "is", "being", "tested"] == result


def test_python_tokenizer_01():
    tokenizer = PythonTokenizer()
    result = tokenizer.tokenize(
        "def my_function(x): return x"
    )
    assert ["my_function", "my", "function", "x"] == result


def test_python_tokenizer_02():
    tokenizer = PythonTokenizer()
    result = tokenizer.tokenize(
        "x = 42"
    )
    assert ["x", "42"] == result


def test_python_tokenizer_03():
    tokenizer = PythonTokenizer()
    SOURCE = '''
        class Example:
            def configure(self):
                if True:
                    return

            def build_extensions(self):
                pass
        '''
    result = tokenizer.tokenize(SOURCE)
    print(f"result: {result}")
    assert isinstance(result, list)
