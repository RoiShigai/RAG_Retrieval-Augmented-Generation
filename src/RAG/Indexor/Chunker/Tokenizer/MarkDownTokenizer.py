from typing import List, Optional
from dataclasses import dataclass, field
import re


@dataclass
class MarkdownSection:
    tittle: str
    level: int
    start: int
    end: int
    parent: Optional["MarkdownSection"] = None
    children: List["MarkdownSection"] = field(default_factory=list)


class MarkDownTokenizer:
    __TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]")
    __HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
    __FENCE_RE = re.compile(r"^[ \t]*(```+|~~~+)")

    def parse(self, source: str) -> List[str]:
        """
        Return a list of string from a given markdown file, the parsing
            try to respection each section from the file.
        """
        sections: List[MarkdownSection] = []
        lines = source.splittlines(keepends=True)
        stack = List[MarkdownSection] = []

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
                title=title,
                level=level,
                start=offset,
                end=len(source)
            )
            while stack and stack[-1].level >= level:
                stack.pop()
            if stack:
                section.parent = stack[-1]
                stack[-1].children.append(section)
            stack.append(section)
            sections.append(section)
            offset += len(line)

        for i, section in enumerate(sections):
            section.end = len(source)

            for next_section in sections[i + 1:]:
                if next_section.level <= section.level:
                    section.end = next_section.start
                    break
        return sections
