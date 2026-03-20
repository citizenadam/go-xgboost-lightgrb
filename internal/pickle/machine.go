package pickle

// machine represents pickle abstract machine with stack, marks and
// memory. It's used to represent unpickled object state while reading pickle file
// command by command
type machine struct {
	stack  []any
	memory map[string]any
	marks  []int
}

func newPickleMachine() *machine {
	return &machine{
		stack:  make([]any, 0),
		memory: make(map[string]any),
	}
}

func (m *machine) pushMark() {
	m.marks = append(m.marks, len(m.stack))
}

func (m *machine) popMark() []any {
	idx := m.marks[len(m.marks)-1]
	m.marks = m.marks[:len(m.marks)-1]
	markedObjects := m.stack[idx:len(m.stack)]
	m.stack = m.stack[:idx]
	return markedObjects
}

func (m *machine) push(obj any) {
	m.stack = append(m.stack, obj)
}

func (m *machine) pop() any {
	obj := m.stack[len(m.stack)-1]
	m.stack = m.stack[:len(m.stack)-1]
	return obj
}

func (m *machine) back() any {
	return m.stack[len(m.stack)-1]
}

func (m *machine) putMemory(key string) {
	m.memory[key] = m.stack[len(m.stack)-1]
}

func (m *machine) getMemory(key string) any {
	return m.memory[key]
}
