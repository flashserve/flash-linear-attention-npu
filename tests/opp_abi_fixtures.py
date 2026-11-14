"""Small ELF metadata fixtures; no executable vendor code is loaded by tests."""
import struct


def write_elf_exports(library, symbols, *, bits=64, byteorder="<"):
    is64 = bits == 64
    header_fmt = byteorder + ("HHIQQQIHHHHHH" if is64 else "HHIIIIIHHHHHH")
    section_fmt = byteorder + ("IIQQQQIIQQ" if is64 else "IIIIIIIIII")
    symbol_fmt = byteorder + ("IBBHQQ" if is64 else "IIIBBH")
    header_size = 16 + struct.calcsize(header_fmt)
    symbol_size = struct.calcsize(symbol_fmt)
    section_size = struct.calcsize(section_fmt)
    strings = bytearray(b"\0")
    entries = bytearray(symbol_size)
    for name in sorted(symbols):
        offset = len(strings)
        strings.extend(name.encode("ascii") + b"\0")
        fields = ((offset, 0x12, 0, 1, 0, 0) if is64 else
                  (offset, 0, 0, 0x12, 0, 1))
        entries.extend(struct.pack(symbol_fmt, *fields))
    strings_offset = header_size + len(entries)
    sections_offset = strings_offset + len(strings)
    ident = b"\x7fELF" + bytes([2 if is64 else 1, 1 if byteorder == "<" else 2, 1]) + bytes(9)
    header = struct.pack(header_fmt, 3, 62 if is64 else 3, 1, 0, 0, sections_offset,
                         0, header_size, 0, 0, section_size, 3, 0)
    sections = bytes(section_size)
    sections += struct.pack(section_fmt, 0, 11, 0, 0, header_size, len(entries), 2, 1, 8, symbol_size)
    sections += struct.pack(section_fmt, 0, 3, 0, 0, strings_offset, len(strings), 0, 0, 1, 0)
    library.parent.mkdir(parents=True, exist_ok=True)
    library.write_bytes(ident + header + entries + strings + sections)
