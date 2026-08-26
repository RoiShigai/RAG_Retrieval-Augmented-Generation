from typing import List
from bisect import bisect_left
from dataclasses import dataclass
import io
import re
import tokenize
import keyword
import textwrap


@dataclass
class PythonToken:
    start: int
    end: int
    value: str


class PythonTokenizer:

    def __init__(self) -> None:
        self.__tokens: List[PythonToken] = []
        self.__token_starts: List[int] = []

    def __is_camel_case(self, s: str) -> bool:
        """ Check if the given string format is CamelCase or not """
        return s != s.lower() and s != s.upper() and "_" not in s

    def __tokenize_dash(self, token: str) -> List[str]:
        """ Split token by dash """
        result: List[str] = []

        for tokens in token.split("-"):
            result.append(tokens)
        return result

    def __tokenize_snake_case(self, token: str) -> List[str]:
        """ Split token by underscore """
        result: List[str] = []

        for tokens in token.split("_"):
            result.append(tokens)
        return result

    def __tokenize_dot(self, token: str) -> List[str]:
        """ Split token by dot """
        result: List[str] = []

        for tokens in token.split("."):
            result.append(tokens)
        return result

    def tokens_for_range(
            self,
            start: int,
            end: int) -> List[str]:
        """
        Return the token corresponding to a range of characters in a file
        """
        index = bisect_left(self.__token_starts, start)
        result = []

        while index < len(self.__tokens):
            token = self.__tokens[index]
            if token.end > end:
                break
            result.append(token.value)
            index += 1
        return result

    def tokenize_file(self, source: str, offsets: List[int]) -> None:
        """
        Return a List of PythonTokens after tokenizing the whole given file

        Parameters:
            source: str | the whole file content
            offsets: List[int] | the offset of each statement from Python ast
        """
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)

        for token in tokens:
            if token.type == tokenize.NAME:
                if not keyword.iskeyword(token.string):
                    values = self._tokenize_identifier(token.string)
                    start = self.__get_offset(
                        offsets,
                        token.start[0],
                        token.start[1]
                    )
                    end = self.__get_offset(
                        offsets,
                        token.end[0],
                        token.end[1]
                    )
                    for value in values:
                        self.__tokens.append(
                            PythonToken(
                                start=start,
                                end=end,
                                value=value
                            )
                        )
                elif token.type == tokenize.NUMBER:
                    start = self.__get_offset(
                        offsets,
                        token.start[0],
                        token.start[1]
                    )
                    end = self.__get_offset(
                        offsets,
                        token.end[0],
                        token.end[1]
                    )
                    self.__tokens.append(
                        PythonToken(
                            start=start,
                            end=end,
                            value=token.string
                        )
                    )
                elif token.type == tokenize.STRING:
                    start = self.__get_offset(
                        offsets,
                        token.start[0],
                        token.start[1]
                    )
                    end = self.__get_offset(
                        offsets,
                        token.end[0],
                        token.end[1]
                    )
                    for value in self.__tokenize_string(token.string):
                        self.__tokens.append(
                            PythonToken(
                                start=start,
                                end=end,
                                value=value
                            )
                        )
        self.__token_starts = [
            token.start
            for token in self.__tokens
        ]

    def tokenize(self, source: str) -> List[str]:
        """
        Return a list of tokens from python's line of code.
        This tokenize function use the python tokenizer to indicate
            the type of token, with that we save only identifier
            string and get rid of the keywords within python code.

        Parameter:
            source | str: the line of code that must be Tokenize
        Return:
            List[str]: list of saved tokens from the line
        """
        result: List[str] = []
        source = textwrap.dedent(source)
        try:
            tokens = tokenize.generate_tokens(
                io.StringIO(source).readline
            )
            for token in tokens:
                if token.type == tokenize.NAME:
                    if not keyword.iskeyword(token.string):
                        result.extend(
                            self._tokenize_identifier(token.string)
                        )
                elif token.type == tokenize.NUMBER:
                    result.append(token.string)
                elif token.type == tokenize.STRING:
                    result.extend(
                        self._tokenize_string(token.string)
                    )
        except IndentationError as e:
                print("\n========== TOKENIZATION ERROR ==========")
                print(repr(e))
                print("========================================\n")
        try:
            tokens = tokenize.tokenize(
                io.BytesIO(source.encode("utf-8")).readline
            )
            for token in tokens:
                pass
            print("TOkenize byte: OK")
        except Exception as e:
            print("Tokenize Bytes error:", repr(e))
        return result

    def _tokenize_identifier(self, identifier: str) -> List[str]:
        """
        Return a list of corresponding token for a given identifier.

        The list will contain atleast the identifier, then we check if
            the identifier format (SnakeCase | CamelCase | ChainCase)
            and tokenize it accordingly of his format.

        Parameter:
            identifier: str | token that will be tokenize

        Return:
            List[str]: List of str containing the identifier and his token.
        """
        result: List[str] = [identifier]

        parts: List[str] = [
            part
            for part in identifier.split("_")
            if part
        ]
        for part in parts:
            result.append(part)
            if self.__is_camel_case(part):
                result.extend(
                    re.findall(
                        r"[A-Z](?:[a-z-0-9]+|[A-Z]*(?=[A-Z]|$))",
                        part
                    )
                )
        return result

    def _tokenize_string(self, string: str) -> List[str]:
        """
        Return a list of corresponding token for a given string.

        The list will contain at least the string, then we check if
            the identifier format (SnakeCase | CamelCase | ChainCase)
            and tokenize it accordingly of his format.

        Parameter:
            string: str | string that will be tokenize

        Return:
            List[str]: List of str containing the string and its token.
        """
        result: List[str] = []

        splitted = string.split()
        for token in splitted:
            result.append(token)
            if '_' in token:
                result.extend(self.__tokenize_snake_case(token))
            if '-' in token:
                result.extend(self.__tokenize_dash(token))
            if '.' in token:
                result.extend(self.__tokenize_dot(token))
            if self.__is_camel_case(token):
                result.extend(
                    re.findall(
                        r"[A-Z](?:[a-z-0-9]+|[A-Z]*(?=[A-Z]|$))",
                        token
                    )
                )
        return result

    def __get_offset(self, offsets: List[int], line: int, column: int) -> int:
        """ Return the char position of the beginning statement """
        return offsets[line - 1] + column
