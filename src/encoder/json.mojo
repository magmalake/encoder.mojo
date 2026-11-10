"""Just enough JSON for safetensors headers and HF config files: a cursor
that reads strings, numbers and int lists, and skips any other value.
"""


struct JsonCursor(Movable):
    var text: String
    var pos: Int

    def __init__(out self, var text: String):
        self.text = text^
        self.pos = 0

    def _at(self, i: Int) -> Int:
        return Int(self.text.as_bytes()[i])

    def _fail(self, what: String) raises:
        raise Error("JSON: ", what, " at byte ", self.pos)

    def skip_ws(mut self):
        var n = self.text.byte_length()
        while self.pos < n:
            var c = self._at(self.pos)
            if c != 32 and c != 9 and c != 10 and c != 13:
                break
            self.pos += 1

    def peek(mut self) raises -> Int:
        self.skip_ws()
        if self.pos >= self.text.byte_length():
            self._fail("unexpected end")
        return self._at(self.pos)

    def expect(mut self, c: Int) raises:
        if self.peek() != c:
            self._fail(String("expected '", chr(c), "'"))
        self.pos += 1

    def try_consume(mut self, c: Int) raises -> Bool:
        if self.peek() == c:
            self.pos += 1
            return True
        return False

    def string(mut self) raises -> String:
        """A string value; handles the common escapes and \\uXXXX in the BMP."""
        self.expect(ord('"'))
        var n = self.text.byte_length()
        var out = String()
        var run = self.pos
        while True:
            if self.pos >= n:
                self._fail("unterminated string")
            var c = self._at(self.pos)
            if c == ord('"'):
                out += String(self.text[byte = run : self.pos])
                self.pos += 1
                return out^
            if c != ord("\\"):
                self.pos += 1
                continue
            out += String(self.text[byte = run : self.pos])
            var e = self._at(self.pos + 1)
            self.pos += 2
            if e == ord("n"):
                out += "\n"
            elif e == ord("t"):
                out += "\t"
            elif e == ord("r"):
                out += "\r"
            elif e == ord("b"):
                out += "\x08"
            elif e == ord("f"):
                out += "\x0c"
            elif e == ord("u"):
                var cp: UInt32 = 0
                for i in range(4):
                    var h = UInt32(self._at(self.pos + i))
                    var d: UInt32
                    if h >= 48 and h <= 57:
                        d = h - 48
                    elif h >= 97 and h <= 102:
                        d = h - 87
                    elif h >= 65 and h <= 70:
                        d = h - 55
                    else:
                        d = 0
                        self._fail("bad \\u escape")
                    cp = cp * 16 + d
                self.pos += 4
                out.write(Codepoint(unsafe_unchecked_codepoint=cp))
            else:  # \" \\ \/
                out += chr(e)
            run = self.pos

    def number(mut self) raises -> Float64:
        self.skip_ws()
        var start = self.pos
        var n = self.text.byte_length()
        while self.pos < n:
            var c = self._at(self.pos)
            if not (
                (c >= 48 and c <= 57)
                or c == ord("-")
                or c == ord("+")
                or c == ord(".")
                or c == ord("e")
                or c == ord("E")
            ):
                break
            self.pos += 1
        if self.pos == start:
            self._fail("expected a number")
        return Float64(String(self.text[byte = start : self.pos]))

    def int(mut self) raises -> Int:
        return Int(self.number())

    def bool(mut self) raises -> Bool:
        self.skip_ws()
        if self.text[byte = self.pos : self.pos + 4] == "true":
            self.pos += 4
            return True
        if self.text[byte = self.pos : self.pos + 5] == "false":
            self.pos += 5
            return False
        self._fail("expected a boolean")
        return False

    def int_list(mut self) raises -> List[Int]:
        var out = List[Int]()
        self.expect(ord("["))
        if self.try_consume(ord("]")):
            return out^
        while True:
            out.append(self.int())
            if self.try_consume(ord("]")):
                return out^
            self.expect(ord(","))

    def skip_value(mut self) raises:
        var c = self.peek()
        if c == ord('"'):
            _ = self.string()
        elif c == ord("{") or c == ord("["):
            var close = ord("}") if c == ord("{") else ord("]")
            self.pos += 1
            if self.try_consume(close):
                return
            while True:
                if c == ord("{"):
                    _ = self.string()
                    self.expect(ord(":"))
                self.skip_value()
                if self.try_consume(close):
                    return
                self.expect(ord(","))
        elif c == ord("t") or c == ord("f"):
            _ = self.bool()
        elif c == ord("n"):
            self.pos += 4
        else:
            _ = self.number()

    def next_key(mut self, mut first: Bool) raises -> Optional[String]:
        """Iterate an object's keys: call after expect('{') with
        first=True; returns None at the closing brace."""
        if first:
            first = False
            if self.try_consume(ord("}")):
                return None
        else:
            if self.try_consume(ord("}")):
                return None
            self.expect(ord(","))
        var key = self.string()
        self.expect(ord(":"))
        return key^
