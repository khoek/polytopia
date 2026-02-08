use hullabaloo_core::AdjacencyList;

#[test]
fn graph_distances_and_diameter_on_cycle4() {
    let graph = AdjacencyList::from_sorted_adjacency_lists(vec![
        vec![1, 3],
        vec![0, 2],
        vec![1, 3],
        vec![0, 2],
    ]);

    let expected = [
        [0usize, 1, 2, 1],
        [1usize, 0, 1, 2],
        [2usize, 1, 0, 1],
        [1usize, 2, 1, 0],
    ];

    for (i, row) in expected.iter().enumerate() {
        for (j, expected_dist) in row.iter().copied().enumerate() {
            assert_eq!(graph.distance(i, j).unwrap(), expected_dist);
        }
    }
    assert_eq!(graph.diameter().unwrap(), 2);
}
