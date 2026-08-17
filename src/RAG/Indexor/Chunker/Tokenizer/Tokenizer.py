from typing import List
import io
import re
import tokenize
import keyword


class PythonTokenizer:

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
            result.append(token)
        return result

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
