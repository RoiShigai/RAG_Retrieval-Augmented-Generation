from typing import List
import io
import re
import tokenize


class PythonTokenizer:

    def __is_camel_case(self, s: str) -> bool:
        """ Check if the given string format is CamelCase or not """
        return s != s.lower() and s != s.upper() and "_" not in s

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
                result.extend(
                    self.tokenize_identifier(token.string)
                )
            elif token.type == tokenize.NUMBER:
                result.append(token.string)
            elif token.type == tokenize.STRING:
                result.append(token.string)
        return result

    def tokenize_identifier(self, identifier: str) -> List[str]:
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
        result: List[str] = []

        result.append(identifier)
        if '_' in identifier:
            splitted = identifier.split("_")
            if len(splitted) > 1:
                result.extend(splitted)
        if '-' in identifier:
            splitted = identifier.split("-")
            if len(splitted) > 1:
                result.extend(splitted)
        if self.__is_camel_case(identifier):
            result.extend(
                re.split(
                    r"[A-Z](?:[a-z-0-9]+|[A-Z]*(?=[A-Z]|$))",
                    identifier
                )
            )
        return result

    def tokenize_string(self, string: str) -> List[str]:
        ...
