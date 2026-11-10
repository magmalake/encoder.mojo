"""Read float32 tensors from a `.safetensors` file.

Format: an 8-byte little-endian header length, a JSON header mapping each
tensor name to {dtype, shape, data_offsets}, then the raw data. Offsets
are relative to the end of the header.
"""

from std.collections import Dict
from std.memory import bitcast
from std.os import stat

from .json import JsonCursor


@fieldwise_init
struct TensorInfo(Copyable, Movable):
    var dtype: String
    var shape: List[Int]
    var begin: Int
    var end: Int


struct SafeTensors(Movable):
    var data: List[UInt8]
    var base: Int
    var tensors: Dict[String, TensorInfo]

    def __init__(out self, path: String) raises:
        # Read with the size known: an unsized read grows its buffer as it
        # goes and costs about three times the file (400 MB of memory for
        # bge-small's 133 MB), which the process doesn't get back.
        var size = Int(stat(path).st_size)
        with open(path, "r") as f:
            self.data = f.read_bytes(size)
        if len(self.data) < 8:
            raise Error(path, ": too short for a safetensors file")
        var n = 0
        for i in range(8):
            n |= Int(self.data[i]) << (8 * i)
        self.base = 8 + n
        if self.base > len(self.data):
            raise Error(path, ": header length past end of file")
        var header = String(from_utf8=Span(self.data)[8 : self.base])
        self.tensors = Dict[String, TensorInfo]()
        var j = JsonCursor(header^)
        j.expect(ord("{"))
        var first = True
        while True:
            var key = j.next_key(first)
            if not key:
                break
            var name = key.value()
            if name == "__metadata__":
                j.skip_value()
                continue
            var dtype = String()
            var shape = List[Int]()
            var offsets = List[Int]()
            j.expect(ord("{"))
            var f2 = True
            while True:
                var k = j.next_key(f2)
                if not k:
                    break
                if k.value() == "dtype":
                    dtype = j.string()
                elif k.value() == "shape":
                    shape = j.int_list()
                elif k.value() == "data_offsets":
                    offsets = j.int_list()
                else:
                    j.skip_value()
            if len(offsets) != 2:
                raise Error(name, ": data_offsets must have 2 entries")
            self.tensors[name] = TensorInfo(
                dtype, shape^, offsets[0], offsets[1]
            )

    def info(self, name: String) raises -> TensorInfo:
        var t = self.tensors.get(name)
        if not t:
            raise Error("no tensor named ", name)
        return t.value().copy()

    def f32(self, name: String, shape: List[Int]) raises -> List[Float32]:
        """The tensor's values, checking it is F32 with the given shape.
        BF16 and F16 tensors are widened to float32."""
        var t = self.info(name)
        if t.shape != shape:
            raise Error(name, ": shape ", t.shape, ", expected ", shape)
        var count = 1
        for d in shape:
            count *= d
        var width: Int
        if t.dtype == "F32":
            width = 4
        elif t.dtype == "BF16" or t.dtype == "F16":
            width = 2
        else:
            raise Error(name, ": unsupported dtype ", t.dtype)
        if t.end - t.begin != count * width:
            raise Error(name, ": byte size does not match its shape")
        var out = List[Float32](capacity=count)
        var p = self.base + t.begin
        if width == 4:
            for i in range(count):
                var q = p + 4 * i
                var bits = (
                    UInt32(self.data[q])
                    | (UInt32(self.data[q + 1]) << 8)
                    | (UInt32(self.data[q + 2]) << 16)
                    | (UInt32(self.data[q + 3]) << 24)
                )
                out.append(bitcast[DType.float32](bits))
        elif t.dtype == "BF16":
            for i in range(count):
                var q = p + 2 * i
                var bits = (UInt32(self.data[q]) << 16) | (
                    UInt32(self.data[q + 1]) << 24
                )
                out.append(bitcast[DType.float32](bits))
        else:
            for i in range(count):
                var q = p + 2 * i
                var bits = UInt16(self.data[q]) | (
                    UInt16(self.data[q + 1]) << 8
                )
                out.append(bitcast[DType.float16](bits).cast[DType.float32]())
        return out^
