from typing import List, Optional
from dataclasses import dataclass, field
import re

from .TokenNormalizer import tokenize_text


@dataclass
class MarkdownSection:
    tittle: str
    level: int
    start: int
    end: int
    parent: Optional["MarkdownSection"] = None
    children: List["MarkdownSection"] = field(default_factory=list)


class MarkDownTokenizer:
    __HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
    __FENCE_RE = re.compile(r"^[ \t]*(```+|~~~+)")

    def tokenize(self, source: str) -> List[str]:
        raw_tokens = tokenize_text(source)
        cleaned_tokens = tokenize_text(self.__clean_markdown(source))
        result = list(raw_tokens)
        for token in cleaned_tokens:
            if token not in result:
                result.append(token)
        return result

    def parse(self, source: str) -> List[MarkdownSection]:
        """
        Return a list of string from a given markdown file, the parsing
            try to respection each section from the file.
        """
        sections: List[MarkdownSection] = []
        lines = source.splitlines(keepends=True)
        stack: List[MarkdownSection] = []

        offset = 0
        in_fence = False
        fence_char: str | None = None

        for line in lines:
            stripped = line.rstrip("\r\n")
            fence_match = self.__FENCE_RE.match(stripped)

            if fence_match:
                fence = fence_match.group(1)
                if not in_fence:
                    in_fence = True
                    fence_char = fence[0]
                elif fence_char == fence[0]:
                    in_fence = False
                    fence_char = None
                offset += len(line)
                continue
            if in_fence:
                offset += len(line)
                continue
            match = self.__HEADING_RE.match(stripped)
            if match is None:
                offset += len(line)
                continue
            level = len(match.group(1))
            title = match.group(2).strip()

            section = MarkdownSection(
                tittle=title,
                level=level,
                start=offset,
                end=len(source)
            )
            while stack and stack[-1].level >= level:
                stack.pop().end = offset
            if stack:
                section.parent = stack[-1]
                stack[-1].children.append(section)
            stack.append(section)
            sections.append(section)
            offset += len(line)

        for section in stack:
            section.end = len(source)
        return sections

    def __clean_markdown(self, source: str) -> str:
        """
        Clean a Markdown content from Markdown vocabulary
        """
        source = re.sub(
            r"^[ \t]*(```+|~~~+)[^\n]*$",
            "",
            source,
            flags=re.MULTILINE,
        )
        source = re.sub(
            r"^[ \t]{0,3}#{1,6}[ \t]+",
            "",
            source,
            flags=re.MULTILINE,
        )
        source = re.sub(
            r"\[([^\]]+)\]\([^)]+\)",
            r"\1",
            source,
        )
        source = re.sub(
            r"!\[([^\]]*)\]\([^)]+\)",
            r"\1",
            source,
        )
        source = re.sub(
            r"`([^`]+)`",
            r"\1",
            source,
        )
        source = re.sub(
            r"\*\*|\*|__|_",
            " ",
            source,
        )
        return source
