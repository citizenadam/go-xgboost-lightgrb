package pickle

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"strconv"
)

// Protocol represents the pickle protocol version
type Protocol int

const (
	Protocol0 Protocol = 0
	Protocol1 Protocol = 1
	Protocol2 Protocol = 2
	Protocol3 Protocol = 3
	Protocol4 Protocol = 4
	Protocol5 Protocol = 5
)

// Decoder implements decoding pickle file from `reader`. It reads op byte
// (pickle command), do command, push result in `machine` object. Main method is
// `Decode`
type Decoder struct {
	reader   io.Reader
	machine  *machine
	protocol Protocol
	// Frame mode (Protocol 4+): indicates we need to read a frame size first
	inFrame bool
	// Buffer for frame data
	frameBuf []byte
}

// NewDecoder creates decoder from io.Reader
func NewDecoder(r io.Reader) *Decoder {
	return &Decoder{
		reader:   r,
		machine:  newPickleMachine(),
		protocol: Protocol0,
		inFrame:  false,
	}
}

// Protocol returns the detected pickle protocol
func (d *Decoder) ProtocolVersion() Protocol {
	return d.protocol
}

// Decode reads pickle commands in loop and return top most result object on the
// machine's stack
func (d *Decoder) Decode() (any, error) {
	nInstr := 0
	buf := make([]byte, 1024)
loop:
	for {
		length, err := d.reader.Read(buf[:1])
		if err != nil {
			return nil, err
		} else if length != 1 {
			return nil, fmt.Errorf("unexpected read")
		}

		nInstr++
		switch buf[0] {
		case opMark:
			d.machine.pushMark()
		case opPut:
			line, err := d.readLine()
			if err != nil {
				return nil, err
			}
			d.machine.putMemory(string(line))
		case opGet:
			line, err := d.readLine()
			if err != nil {
				return nil, err
			}
			obj := d.machine.getMemory(string(line))
			d.machine.push(obj)
		case opGlobal:
			module, err := d.readLine()
			if err != nil {
				return nil, err
			}
			name, err := d.readLine()
			if err != nil {
				return nil, err
			}
			d.machine.push(Global{Module: string(module), Name: string(name)})
		case opLong:
			line, err := d.readLine()
			if err != nil {
				return nil, err
			}
			l := len(line)
			if l < 2 || line[l-1] != 'L' {
				return nil, fmt.Errorf("unexpected long format: %s", line)
			}
			v, err := strconv.Atoi(string(line[:l-1]))
			if err != nil {
				return nil, fmt.Errorf("unexpected long format: %s", line)
			}
			d.machine.push(v)
		case opInt:
			line, err := d.readLine()
			if err != nil {
				return nil, err
			}
			v, err := strconv.Atoi(string(line))
			if err != nil {
				return nil, err
			}
			d.machine.push(v)
		case opFloat:
			line, err := d.readLine()
			if err != nil {
				return nil, err
			}
			v, err := strconv.ParseFloat(string(line), 64)
			if err != nil {
				return nil, err
			}
			d.machine.push(v)
		case opTuple:
			tuple := append(Tuple{}, d.machine.popMark()...)
			d.machine.push(tuple)
		case opList:
			list := append(List{}, d.machine.popMark()...)
			d.machine.push(list)
		case opAppend:
			el := d.machine.pop()
			obj := d.machine.pop()
			list, err := toList(obj, -1)
			if err != nil {
				return nil, err
			}
			list = append(list, el)
			d.machine.push(list)
		case opUnicode:
			line, err := d.readLine()
			if err != nil {
				return nil, err
			}
			d.machine.push(decodeUnicode(line))
		case opReduce:
			args := d.machine.pop()
			callable := d.machine.pop()
			r := Reduce{Callable: callable}
			if t, ok := args.(Tuple); ok {
				r.Args = t
			} else {
				return nil, fmt.Errorf("reduce: unexpected args %v", args)
			}
			d.machine.push(r)
		case opNone:
			d.machine.push(None{})
		case opBuild:
			args := d.machine.pop()
			obj := d.machine.pop()
			b := Build{Object: obj, Args: args}
			d.machine.push(b)
		case opDict:
			objs := d.machine.popMark()
			nObjs := len(objs)
			dict := make(Dict, 0)
			if nObjs%2 != 0 {
				return nil, fmt.Errorf("dict: expected event number of objects (got %d)", nObjs)
			}
			for i := 0; i < nObjs; i += 2 {
				unicode, err := toUnicode(objs[i], -1)
				if err != nil {
					return nil, err
				}
				key := string(unicode)
				dict[key] = objs[i+1]
			}
			d.machine.push(dict)
		case opSetitem:
			v := d.machine.pop()
			k := d.machine.pop()
			dict, err := toDict(d.machine.back())
			if err != nil {
				return nil, err
			}
			unicode, err := toUnicode(k, -1)
			if err != nil {
				return nil, err
			}
			key := string(unicode)
			dict[key] = v
		case opStop:
			break loop

		// Protocol 1 binary opcodes
		case opBinInt:
			b := make([]byte, 4)
			_, err := io.ReadFull(d.reader, b)
			if err != nil {
				return nil, fmt.Errorf("BININT: %w", err)
			}
			d.machine.push(int(binary.LittleEndian.Uint32(b)))
		case opBinInt1:
			b := make([]byte, 1)
			_, err := io.ReadFull(d.reader, b)
			if err != nil {
				return nil, fmt.Errorf("BININT1: %w", err)
			}
			d.machine.push(int(b[0]))
		case opBinInt2:
			b := make([]byte, 2)
			_, err := io.ReadFull(d.reader, b)
			if err != nil {
				return nil, fmt.Errorf("BININT2: %w", err)
			}
			d.machine.push(int(binary.LittleEndian.Uint16(b)))
		case opBinFloat:
			b := make([]byte, 8)
			_, err := io.ReadFull(d.reader, b)
			if err != nil {
				return nil, fmt.Errorf("BINFLOAT: %w", err)
			}
			d.machine.push(math.Float64frombits(binary.LittleEndian.Uint64(b)))
		case opBinString, opShortBinString:
			var n int
			if buf[0] == opBinString {
				b := make([]byte, 4)
				_, err := io.ReadFull(d.reader, b)
				if err != nil {
					return nil, fmt.Errorf("BINSTRING: %w", err)
				}
				n = int(binary.LittleEndian.Uint32(b))
			} else {
				b := make([]byte, 1)
				_, err := io.ReadFull(d.reader, b)
				if err != nil {
					return nil, fmt.Errorf("SHORT_BINSTRING: %w", err)
				}
				n = int(b[0])
			}
			s := make([]byte, n)
			_, err := io.ReadFull(d.reader, s)
			if err != nil {
				return nil, fmt.Errorf("BINSTRING data: %w", err)
			}
			d.machine.push(string(s))
		case opBinUnicode:
			b := make([]byte, 4)
			_, err := io.ReadFull(d.reader, b)
			if err != nil {
				return nil, fmt.Errorf("BINUNICODE: %w", err)
			}
			n := int(binary.LittleEndian.Uint32(b))
			s := make([]byte, n)
			_, err = io.ReadFull(d.reader, s)
			if err != nil {
				return nil, fmt.Errorf("BINUNICODE data: %w", err)
			}
			d.machine.push(Unicode(s))
		case opEmptyList:
			d.machine.push(List{})
		case opEmptyDict:
			d.machine.push(Dict{})
		case opEmptyTuple:
			d.machine.push(Tuple{})
		case opPut2:
			b := make([]byte, 1)
			_, err := io.ReadFull(d.reader, b)
			if err != nil {
				return nil, fmt.Errorf("PUT2: %w", err)
			}
			d.machine.putMemory(string(b))
		case opGet1:
			b := make([]byte, 1)
			_, err := io.ReadFull(d.reader, b)
			if err != nil {
				return nil, fmt.Errorf("GET1: %w", err)
			}
			obj := d.machine.getMemory(string(b))
			d.machine.push(obj)

		// Protocol 2 opcodes
		case opProto:
			b := make([]byte, 1)
			_, err := io.ReadFull(d.reader, b)
			if err != nil {
				return nil, fmt.Errorf("PROTO: %w", err)
			}
			d.protocol = Protocol(b[0])
			if d.protocol < 0 || d.protocol > 5 {
				return nil, fmt.Errorf("unsupported pickle protocol: %d", d.protocol)
			}
		case opNewObj:
			args := d.machine.pop()
			callable := d.machine.pop()
			r := Reduce{Callable: callable}
			if t, ok := args.(Tuple); ok {
				r.Args = t
			} else {
				return nil, fmt.Errorf("NEWOBJ: unexpected args %v", args)
			}
			d.machine.push(r)
		case opNewTrue:
			d.machine.push(true)
		case opNewFalse:
			d.machine.push(false)
		case opLong1:
			b := make([]byte, 1)
			_, err := io.ReadFull(d.reader, b)
			if err != nil {
				return nil, fmt.Errorf("LONG1: %w", err)
			}
			n := int(b[0])
			data := make([]byte, n)
			_, err = io.ReadFull(d.reader, data)
			if err != nil {
				return nil, fmt.Errorf("LONG1 data: %w", err)
			}
			// Convert to int (ignoring overflow for our use case)
			v := int64(0)
			for _, b := range data {
				v = v<<8 | int64(b)
			}
			d.machine.push(int(v))

		// Protocol 3 opcodes (bytes)
		case opBinBytes:
			b := make([]byte, 4)
			_, err := io.ReadFull(d.reader, b)
			if err != nil {
				return nil, fmt.Errorf("BINBYTES: %w", err)
			}
			n := int(binary.LittleEndian.Uint32(b))
			data := make([]byte, n)
			_, err = io.ReadFull(d.reader, data)
			if err != nil {
				return nil, fmt.Errorf("BINBYTES data: %w", err)
			}
			d.machine.push(data)
		case opBinBytes1:
			b := make([]byte, 1)
			_, err := io.ReadFull(d.reader, b)
			if err != nil {
				return nil, fmt.Errorf("BINBYTES1: %w", err)
			}
			n := int(b[0])
			data := make([]byte, n)
			_, err = io.ReadFull(d.reader, data)
			if err != nil {
				return nil, fmt.Errorf("BINBYTES1 data: %w", err)
			}
			d.machine.push(data)

		// Protocol 4 opcodes
		case opFrame:
			b := make([]byte, 8)
			_, err := io.ReadFull(d.reader, b)
			if err != nil {
				return nil, fmt.Errorf("FRAME: %w", err)
			}
			_ = binary.LittleEndian.Uint64(b) // frame size - not used in current implementation
		case opShortBinUnicode:
			b := make([]byte, 1)
			_, err := io.ReadFull(d.reader, b)
			if err != nil {
				return nil, fmt.Errorf("SHORT_BINUNICODE: %w", err)
			}
			n := int(b[0])
			s := make([]byte, n)
			_, err = io.ReadFull(d.reader, s)
			if err != nil {
				return nil, fmt.Errorf("SHORT_BINUNICODE data: %w", err)
			}
			d.machine.push(Unicode(s))
		case opBinUnicode8:
			b := make([]byte, 8)
			_, err := io.ReadFull(d.reader, b)
			if err != nil {
				return nil, fmt.Errorf("BINUNICODE8: %w", err)
			}
			n := int(binary.LittleEndian.Uint64(b))
			s := make([]byte, n)
			_, err = io.ReadFull(d.reader, s)
			if err != nil {
				return nil, fmt.Errorf("BINUNICODE8 data: %w", err)
			}
			d.machine.push(Unicode(s))
		case opBinBytes8:
			b := make([]byte, 8)
			_, err := io.ReadFull(d.reader, b)
			if err != nil {
				return nil, fmt.Errorf("BINBYTES8: %w", err)
			}
			n := int(binary.LittleEndian.Uint64(b))
			data := make([]byte, n)
			_, err = io.ReadFull(d.reader, data)
			if err != nil {
				return nil, fmt.Errorf("BINBYTES8 data: %w", err)
			}
			d.machine.push(data)
		case opMemoize:
			d.machine.putMemory(fmt.Sprintf("%d", len(d.machine.memory)))
		case opStackGlobal:
			name := d.machine.pop()
			module := d.machine.pop()
			moduleStr, ok := module.(string)
			if !ok {
				return nil, fmt.Errorf("STACK_GLOBAL: expected module string, got %T", module)
			}
			nameStr, ok := name.(string)
			if !ok {
				return nil, fmt.Errorf("STACK_GLOBAL: expected name string, got %T", name)
			}
			d.machine.push(Global{Module: moduleStr, Name: nameStr})
		case opAddItems:
			items := d.machine.popMark()
			_ = d.machine.pop() // set - not used
			// For our purposes, just push items back as a list
			d.machine.push(items)
		case opFrozenset:
			items := d.machine.popMark()
			d.machine.push(items)
		case opNewObjEx:
			kw := d.machine.pop()
			args := d.machine.pop()
			callable := d.machine.pop()
			r := Reduce{Callable: callable}
			if t, ok := args.(Tuple); ok {
				r.Args = t
			}
			// Store kwargs in a special way for now, just include in args
			if kw != nil {
				r.Args = append(r.Args, kw)
			}
			d.machine.push(r)

		// Protocol 5 opcodes
		case opByteArray8:
			b := make([]byte, 8)
			_, err := io.ReadFull(d.reader, b)
			if err != nil {
				return nil, fmt.Errorf("BYTEARRAY8: %w", err)
			}
			n := int(binary.LittleEndian.Uint64(b))
			data := make([]byte, n)
			_, err = io.ReadFull(d.reader, data)
			if err != nil {
				return nil, fmt.Errorf("BYTEARRAY8 data: %w", err)
			}
			// Convert to string for compatibility
			d.machine.push(string(data))
		case opNextBuffer, opReadOnlyBuffer:
			// Out-of-band buffers - not typically used in tree models
			// Push nil as placeholder
			d.machine.push(nil)

		default:
			return nil, fmt.Errorf("unknown op code: %d ('%c')", buf[0], buf[0])
		}
	}
	return d.machine.back(), nil
}

func (d *Decoder) readLine() ([]byte, error) {
	line := make([]byte, 0)
	buf := make([]byte, 1)
	for {
		len, err := d.reader.Read(buf)
		if err != nil {
			return nil, err
		} else if len != 1 {
			return nil, fmt.Errorf("unexpected read")
		}
		if buf[0] == '\n' {
			break
		}
		line = append(line, buf[0])
	}
	return line, nil
}

func decodeUnicode(rawString []byte) Unicode {
	// \u000a == \n
	ret := bytes.ReplaceAll(rawString, []byte{'\\', 'u', '0', '0', '0', 'a'}, []byte{'\n'})
	// \u005c' == \\
	ret = bytes.ReplaceAll(ret, []byte{'\\', 'u', '0', '0', '5', 'c'}, []byte{'\\'})
	return Unicode(ret)
}

// Codes below are taken from https://github.com/kisielk/og-rek
// Thanks to authors!
// Opcodes
const (
	// Protocol 0

	opMark    byte = '(' // push special markobject on stack
	opStop    byte = '.' // every pickle ends with STOP
	opPop     byte = '0' // discard topmost stack item
	opDup     byte = '2' // duplicate top stack item
	opFloat   byte = 'F' // push float object; decimal string argument
	opInt     byte = 'I' // push integer or bool; decimal string argument
	opLong    byte = 'L' // push long; decimal string argument
	opNone    byte = 'N' // push None
	opPersid  byte = 'P' // push persistent object; id is taken from string arg
	opReduce  byte = 'R' // apply callable to argtuple, both on stack
	opString  byte = 'S' // push string; NL-terminated string argument
	opUnicode byte = 'V' // push Unicode string; raw-unicode-escaped"d argument
	opAppend  byte = 'a' // append stack top to list below it
	opBuild   byte = 'b' // call __setstate__ or __dict__.update()
	opGlobal  byte = 'c' // push self.find_class(modname, name); 2 string args
	opDict    byte = 'd' // build a dict from stack items
	opGet     byte = 'g' // push item from memo on stack; index is string arg
	opInst    byte = 'i' // build & push class instance
	opList    byte = 'l' // build list from topmost stack items
	opPut     byte = 'p' // store stack top in memo; index is string arg
	opSetitem byte = 's' // add key+value pair to dict
	opTuple   byte = 't' // build tuple from topmost stack items

	// Protocol 1

	opPut2           byte = 'q'  // store stack top in memo; index is 1-byte arg
	opGet1           byte = 'h'  // push item from memo on stack; index is 1-byte arg
	opBinInt         byte = 'J'  // push 4-byte signed int
	opBinInt1        byte = 'K'  // push 1-byte unsigned int
	opBinInt2        byte = 'M'  // push 2-byte unsigned int
	opBinFloat       byte = 'G'  // push 8-byte float
	opBinString      byte = 'T'  // push string; counted binary string argument
	opShortBinString byte = 'U'  // push string; counted binary string argument < 256 bytes
	opBinUnicode     byte = 'X'  // push Unicode string; counted binary argument
	opBinPersistID   byte = 'Q'  // push persistent object; id is taken from 1-byte arg
	opEmptyList      byte = ']'  // push empty list
	opEmptyDict      byte = '}'  // push empty dict
	opEmptyTuple     byte = ')'  // push empty tuple
	opBinGet         byte = 'j'  // push item from memo; index is 4-byte arg
	opLong1          byte = 0x8a // push long; 1-byte length

	// Protocol 2

	opProto    byte = 0x80 // protocol version indicator
	opNewObj   byte = 0x81 // build object using __reduce_ex__
	opExt1     byte = 0x82 // push object; 1-byte extension index
	opExt2     byte = 0x83 // push object; 2-byte extension index
	opExt4     byte = 0x84 // push object; 4-byte extension index
	opNewTrue  byte = 0x88 // push True
	opNewFalse byte = 0x89 // push False
	opUnicode1 byte = 0x8a // push Unicode string; 1-byte length
	opLong4    byte = 0x8b // push really big long

	// Protocol 3

	opBinBytes  byte = 'B'  // push bytes; counted binary string argument
	opBinBytes1 byte = 0x8f // push bytes; 1-byte length

	// Protocol 4

	opShortBinUnicode byte = 0x8c // push short string; UTF-8 length < 256 bytes
	opBinUnicode8     byte = 0x8d // push very long string
	opBinBytes8       byte = 0x8e // push very long bytes string
	opSetSimple       byte = 0x8b // set item with simple key
	opAddItems        byte = 0x90 // modify set by adding items
	opFrozenset       byte = 0x91 // push empty frozenset
	opNewObjEx        byte = 0x92 // build object with kwargs
	opStackGlobal     byte = 0x93 // same as GLOBAL but using names from stack
	opMemoize         byte = 0x94 // store top of stack in memo
	opFrame           byte = 0x95 // indicate beginning of a new frame

	// Protocol 5

	opByteArray8     byte = 0x96 // push bytearray
	opNextBuffer     byte = 0x97 // push next out-of-band buffer
	opReadOnlyBuffer byte = 0x98 // make buffer readonly
)
