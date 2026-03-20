package leaves

import "testing"

func TestGetNLeavesEmpty(t *testing.T) {
	trees := []lgTree{}
	result := GetNLeaves(trees)
	if len(result) != 0 {
		t.Errorf("GetNLeaves([]) length = %d, want 0", len(result))
	}
}

func TestGetNLeavesSingleTree(t *testing.T) {
	// Single tree with 0 nodes -> 1 leaf
	trees := []lgTree{{}}
	result := GetNLeaves(trees)
	if len(result) != 1 {
		t.Errorf("GetNLeaves([single]) length = %d, want 1", len(result))
	}
	if result[0] != 1 {
		t.Errorf("GetNLeaves([single])[0] = %d, want 1", result[0])
	}

	// Single tree with 2 nodes -> 3 leaves
	trees = []lgTree{{nodes: []lgNode{{}, {}}}}
	result = GetNLeaves(trees)
	if len(result) != 1 {
		t.Errorf("GetNLeaves([two nodes]) length = %d, want 1", len(result))
	}
	if result[0] != 3 {
		t.Errorf("GetNLeaves([two nodes])[0] = %d, want 3", result[0])
	}
}

func TestGetNLeavesMultipleTrees(t *testing.T) {
	// Three trees: 0 nodes, 1 node, 2 nodes
	trees := []lgTree{
		{},                        // 0 nodes -> 1 leaf
		{nodes: []lgNode{{}}},     // 1 node -> 2 leaves
		{nodes: []lgNode{{}, {}}}, // 2 nodes -> 3 leaves
	}
	result := GetNLeaves(trees)
	if len(result) != 3 {
		t.Errorf("GetNLeaves([0,1,2 nodes]) length = %d, want 3", len(result))
	}
	if result[0] != 1 {
		t.Errorf("GetNLeaves([0,1,2 nodes])[0] = %d, want 1", result[0])
	}
	if result[1] != 2 {
		t.Errorf("GetNLeaves([0,1,2 nodes])[1] = %d, want 2", result[1])
	}
	if result[2] != 3 {
		t.Errorf("GetNLeaves([0,1,2 nodes])[2] = %d, want 3", result[2])
	}
}
