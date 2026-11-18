"""Exercise SAVEMONEY's assignment constraints with the installed PuLP/CBC."""

import unittest

import numpy as np

from savemoney.pre_survey import pre_survey_core as psc


class TestPuLPCompatibility(unittest.TestCase):
    def assert_valid_assignment(self, assignment, clusters, group_sizes):
        self.assertIsNotNone(assignment)
        self.assertEqual(assignment.shape, (len(clusters), len(group_sizes)))
        seen = []
        for row, cluster in zip(assignment, clusters):
            members = [int(value) for value in row if value >= 0]
            self.assertCountEqual(members, cluster)
            seen.extend(members)
        self.assertCountEqual(seen, range(sum(map(len, clusters))))
        np.testing.assert_array_equal((assignment >= 0).sum(axis=0), group_sizes)

    def test_balanced_assignment_preserves_cluster_constraints(self):
        clusters = [[0, 1], [2], [3]]
        for group_count in (3, np.int64(3)):
            with self.subTest(group_count_type=type(group_count).__name__):
                solver = psc.PlasmidAssignmentSolver(group_count, clusters, [])

                assignment = solver.generate_example()

                self.assert_valid_assignment(assignment, clusters, [2, 1, 1])

    def test_forbidden_pair_is_separated(self):
        clusters = [[0, 1], [2, 3]]
        solver = psc.PlasmidAssignmentSolver(2, clusters, [])
        solver.idx_pairs_removed = [(0, 2)]

        assignment = solver.generate_example()

        self.assert_valid_assignment(assignment, clusters, [2, 2])
        group_0 = int(np.where(assignment == 0)[1][0])
        group_2 = int(np.where(assignment == 2)[1][0])
        self.assertNotEqual(group_0, group_2)

    def test_incompatible_forbidden_pairs_are_infeasible(self):
        solver = psc.PlasmidAssignmentSolver(2, [[0, 1], [2, 3]], [])
        solver.idx_pairs_removed = [(0, 2), (0, 3)]

        self.assertIsNone(solver.generate_example())

    def test_picross_preserves_row_and_column_counts(self):
        candidates = psc.PlasmidAssignment.solve_simple_picross(5, 3, [2, 2, 1])

        self.assertEqual(len(candidates), 1)
        assignment = candidates[0]
        self.assertEqual(assignment.shape, (3, 3))
        self.assertTrue(np.isin(assignment, [0, 1]).all())
        np.testing.assert_array_equal(assignment.sum(axis=1), [2, 2, 1])
        np.testing.assert_array_equal(assignment.sum(axis=0), [2, 2, 1])


if __name__ == "__main__":
    unittest.main()
