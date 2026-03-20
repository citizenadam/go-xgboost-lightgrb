package pickle

import (
	"testing"
)

func TestNewPickleMachine(t *testing.T) {
	m := newPickleMachine()
	if m == nil {
		t.Error("newPickleMachine() returned nil")
	}
	if len(m.stack) != 0 {
		t.Errorf("Expected empty stack, got %v", m.stack)
	}
	if len(m.marks) != 0 {
		t.Errorf("Expected empty marks, got %v", m.marks)
	}
	if m.memory == nil {
		t.Error("Expected memory map to be initialized")
	} else if len(m.memory) != 0 {
		t.Errorf("Expected empty memory, got %v", m.memory)
	}
}

func TestPushPop(t *testing.T) {
	m := newPickleMachine()

	// Test pushing and popping integers
	m.push(1)
	m.push(2)
	m.push(3)

	if len(m.stack) != 3 {
		t.Errorf("Expected stack length 3, got %d", len(m.stack))
	}

	obj := m.pop()
	if obj != 3 {
		t.Errorf("Expected popped value 3, got %v", obj)
	}
	if len(m.stack) != 2 {
		t.Errorf("Expected stack length 2 after pop, got %d", len(m.stack))
	}

	obj = m.pop()
	if obj != 2 {
		t.Errorf("Expected popped value 2, got %v", obj)
	}
	if len(m.stack) != 1 {
		t.Errorf("Expected stack length 1 after pop, got %d", len(m.stack))
	}

	obj = m.pop()
	if obj != 1 {
		t.Errorf("Expected popped value 1, got %v", obj)
	}
	if len(m.stack) != 0 {
		t.Errorf("Expected stack length 0 after pop, got %d", len(m.stack))
	}

	// Test popping from empty stack (should panic)
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic when popping from empty stack")
		}
	}()
	m.pop()
}

func TestPushPopMark(t *testing.T) {
	m := newPickleMachine()

	// Test marking and popping marked objects
	m.push("a")
	m.push("b")
	m.pushMark()
	m.push("c")
	m.push("d")
	m.push("e")

	marked := m.popMark()
	if len(marked) != 3 {
		t.Errorf("Expected 3 marked objects, got %d", len(marked))
	}
	if marked[0] != "c" || marked[1] != "d" || marked[2] != "e" {
		t.Errorf("Expected marked objects [c d e], got %v", marked)
	}
	if len(m.stack) != 2 {
		t.Errorf("Expected stack length 2 after popMark, got %d", len(m.stack))
	}
	if m.stack[0] != "a" || m.stack[1] != "b" {
		t.Errorf("Expected remaining stack [a b], got %v", m.stack)
	}

	// Test popMark with no marks (should panic)
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic when popping mark with empty marks")
		}
	}()
	m.popMark()
}

func TestPutGetMemory(t *testing.T) {
	m := newPickleMachine()

	// Test putting and getting from memory
	m.push(42)
	m.putMemory("answer")

	value := m.getMemory("answer")
	if value != 42 {
		t.Errorf("Expected memory value 42, got %v", value)
	}

	// Test getting non-existent key
	value = m.getMemory("nonexistent")
	if value != nil {
		t.Errorf("Expected nil for nonexistent key, got %v", value)
	}

	// Test overwriting memory
	m.push("hello")
	m.putMemory("answer")
	value = m.getMemory("answer")
	if value != "hello" {
		t.Errorf("Expected memory value 'hello', got %v", value)
	}
}

func TestBack(t *testing.T) {
	m := newPickleMachine()

	// Test back() on empty stack (should panic)
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic when calling back() on empty stack")
		}
	}()
	m.back()

	// Test back() with values
	m.push(1)
	m.push(2)
	m.push(3)

	if m.back() != 3 {
		t.Errorf("Expected back() to return 3, got %v", m.back())
	}

	// Verify stack is unchanged
	if len(m.stack) != 3 {
		t.Errorf("Expected stack length unchanged after back(), got %d", len(m.stack))
	}
	if m.stack[0] != 1 || m.stack[1] != 2 || m.stack[2] != 3 {
		t.Errorf("Expected stack [1 2 3] after back(), got %v", m.stack)
	}
}
