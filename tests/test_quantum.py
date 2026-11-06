import unittest
import numpy as np

from statevector_sim import StatevectorSimulator
from circuit import Circuit
from hamiltonian import Hamiltonian
from vqe import run_vqe, vqe_cost


class SimulatorTests(unittest.TestCase):
    def test_single_qubit_gates(self):
        for n in (1, 2, 3, 4):
            for q in range(n):
                sim = StatevectorSimulator(n)
                sim.apply_gate(sim.X, (q,))
                expected = np.zeros(1 << n, dtype=complex)
                expected[1 << (n - 1 - q)] = 1.0
                np.testing.assert_allclose(sim.state, expected, atol=1e-13)
                sim.apply_gate(sim.H, (q,))
                self.assertAlmostEqual(np.linalg.norm(sim.state), 1.0)

    def test_all_cnot_basis_states(self):
        for n in (2, 3, 4):
            for control in range(n):
                for target in range(n):
                    if control == target:
                        continue
                    cm = 1 << (n - 1 - control)
                    tm = 1 << (n - 1 - target)
                    for i in range(1 << n):
                        state = np.eye(1 << n, dtype=complex)[i]
                        sim = StatevectorSimulator(n, state)
                        sim.cnot(control, target)
                        expected_idx = i ^ tm if i & cm else i
                        np.testing.assert_allclose(
                            sim.state, np.eye(1 << n)[expected_idx], atol=1e-13)

    def test_swap_reversed_and_arbitrary_pairs(self):
        for n in (2, 3, 4):
            for a in range(n):
                for b in range(n):
                    for i in range(1 << n):
                        sim = StatevectorSimulator(n, np.eye(1 << n)[i])
                        sim.swap_qubits(a, b)
                        ma, mb = 1 << (n - 1 - a), 1 << (n - 1 - b)
                        expected_idx = i ^ ma ^ mb if bool(i & ma) != bool(i & mb) else i
                        np.testing.assert_allclose(sim.state, np.eye(1 << n)[expected_idx])

    def test_entanglement_and_replay(self):
        c = Circuit(2)
        c.h(0)
        c.cnot(0, 1)
        expected = np.array([1, 0, 0, 1], complex) / np.sqrt(2)
        np.testing.assert_allclose(c.sim.state, expected)
        c.run()
        np.testing.assert_allclose(c.sim.state, expected)
        np.testing.assert_allclose(c.density_matrix(), np.outer(expected, expected.conj()))

    def test_validation(self):
        with self.assertRaises(ValueError):
            StatevectorSimulator(0)
        with self.assertRaises(ValueError):
            StatevectorSimulator(2).cnot(0, 0)
        with self.assertRaises(ValueError):
            StatevectorSimulator(2).cnot(-1, 1)
        with self.assertRaises(ValueError):
            StatevectorSimulator(2).apply_gate(np.eye(4), (0,))
        with self.assertRaises(ValueError):
            StatevectorSimulator(1).measure_all(0)
        with self.assertRaises(ValueError):
            StatevectorSimulator.kraus_operators('dephasing', 2)


class HamiltonianTests(unittest.TestCase):
    def setUp(self):
        self.ham = Hamiltonian()

    def test_exact_h2_energy(self):
        self.assertAlmostEqual(self.ham.exact_ground_energy(), -1.13730604020238, places=9)
        np.testing.assert_allclose(self.ham.matrix(), self.ham.matrix().conj().T)

    def test_pauli_endianness(self):
        # |01>: q0 is 0 and q1 is 1
        sim = StatevectorSimulator(2, np.array([0, 1, 0, 0], complex))
        self.assertEqual(self.ham._measure_pauli(sim, 'Z0', 200), 1.0)
        self.assertEqual(self.ham._measure_pauli(sim, 'Z1', 200), -1.0)
        self.assertEqual(self.ham._measure_pauli(sim, 'Z0Z1', 200), -1.0)

    def test_x_and_y_basis_measurements(self):
        plus = np.array([1, 1], complex) / np.sqrt(2)
        plus_y = np.array([1, 1j], complex) / np.sqrt(2)
        simx = StatevectorSimulator(1, plus)
        simy = StatevectorSimulator(1, plus_y)
        self.assertEqual(self.ham._measure_pauli(simx, 'X0', 300), 1.0)
        self.assertEqual(self.ham._measure_pauli(simy, 'Y0', 300), 1.0)
        bell = StatevectorSimulator(2, np.array([1, 0, 0, 1], complex) / np.sqrt(2))
        self.assertEqual(self.ham._measure_pauli(bell, 'X0X1', 300), 1.0)
        self.assertEqual(self.ham._measure_pauli(bell, 'Y0Y1', 300), -1.0)

    def test_pauli_matrices(self):
        for term in ('I', 'X0', 'Y1', 'Z0Z1', 'X0Y1'):
            matrix = self.ham.pauli_matrix(term)
            np.testing.assert_allclose(matrix @ matrix, np.eye(4))
            np.testing.assert_allclose(matrix, matrix.conj().T)
        with self.assertRaises(ValueError):
            self.ham.pauli_matrix('X0Z0')
        with self.assertRaises(ValueError):
            self.ham.pauli_matrix('X4')

    def test_mixed_pauli_sampled_expectations(self):
        np.random.seed(91)
        rng = np.random.default_rng(91)
        ham = Hamiltonian()
        for n, term in ((2, 'X0Y1'), (3, 'Z0X1Y2'), (4, 'Y0Z2X3')):
            vec = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
            vec /= np.linalg.norm(vec)
            sim = StatevectorSimulator(n, vec)
            exact = np.vdot(vec, ham.pauli_matrix(term, n) @ vec).real
            measured = ham._measure_pauli(sim, term, 50000)
            self.assertLess(abs(measured - exact), 0.016)

    def test_sampled_and_exact_energy(self):
        np.random.seed(21)
        rng = np.random.default_rng(8)
        for _ in range(5):
            vec = rng.normal(size=4) + 1j * rng.normal(size=4)
            vec /= np.linalg.norm(vec)
            sim = StatevectorSimulator(2, vec)
            exact = self.ham.compute_energy_exact(sim)
            measured = self.ham.compute_energy(sim, shots=60000)
            rho = np.outer(vec, vec.conj())
            from_rho = self.ham.compute_energy_density(rho)
            self.assertAlmostEqual(exact, from_rho, places=11)
            self.assertLess(abs(measured - exact), 0.012)


class NoiseTests(unittest.TestCase):
    def test_kraus_completeness(self):
        for channel in ('depolarizing', 'dephasing', 'amplitude_damping'):
            for p in (0, 0.2, 0.8, 1):
                Ks = StatevectorSimulator.kraus_operators(channel, p)
                total = sum(K.conj().T @ K for K in Ks)
                np.testing.assert_allclose(total, np.eye(2), atol=1e-14)

    def test_density_matrix_amplitude_damping(self):
        c = Circuit(2)
        c.x(0)
        c.apply_noise('amplitude_damping', 0.3)
        rho = c.density_matrix()
        np.testing.assert_allclose(np.diag(rho), [0.3, 0, 0.7, 0], atol=1e-12)
        self.assertAlmostEqual(np.trace(rho).real, 1)

    def test_density_matrix_dephasing(self):
        c = Circuit(2)
        c.h(0)
        c.apply_noise('dephasing', 0.3)
        rho = c.density_matrix()
        pauli = Hamiltonian().pauli_matrix('X0')
        self.assertAlmostEqual(np.trace(rho @ pauli).real, 0.4, places=12)

    def test_density_matrix_is_valid(self):
        c = Circuit(2)
        c.h(0)
        c.cnot(0, 1)
        c.apply_noise('depolarizing', 0.12)
        c.apply_noise('amplitude_damping', 0.05)
        rho = c.density_matrix()
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-13)
        self.assertAlmostEqual(np.trace(rho).real, 1)
        self.assertGreaterEqual(np.min(np.linalg.eigvalsh(rho)), -1e-12)

    def test_trajectories_vs_exact_channel(self):
        np.random.seed(18)
        c = Circuit(1)
        c.h(0)
        c.apply_noise('dephasing', 0.35)
        target = c.density_matrix()
        empirical = np.zeros((2, 2), dtype=complex)
        for _ in range(2000):
            c.run()
            empirical += np.outer(c.sim.state, c.sim.state.conj())
        empirical /= 2000
        np.testing.assert_allclose(empirical, target, atol=0.045)


class VQETests(unittest.TestCase):
    def test_zero_noise_matches_ideal(self):
        theta = np.array([0.2, -0.7, 0.3, 0.4])
        ideal = Circuit(2)
        ideal.h2_ansatz_ideal(theta)
        noisy = Circuit(2)
        noisy.h2_ansatz_noisy(theta, [{'channel': 'depolarizing', 'p': 0},
                                      {'channel': 'amplitude_damping', 'p': 0}])
        np.testing.assert_allclose(noisy.sim.state, ideal.sim.state, atol=1e-12)
        np.testing.assert_allclose(noisy.density_matrix(),
                                   np.outer(ideal.sim.state, ideal.sim.state.conj()), atol=1e-12)

    def test_ansatz_matches_density_matrix(self):
        c = Circuit(2)
        c.h2_ansatz_ideal([0.2, -0.7, 0.3, 0.4])
        psi = c.sim.state
        np.testing.assert_allclose(c.density_matrix(), np.outer(psi, psi.conj()))
        self.assertEqual(len(c.operations), 5)
        c.h2_ansatz_ideal([0.3, 0.2, -0.4, 0.6])
        self.assertEqual(len(c.operations), 5)

    def test_ideal_optimization(self):
        ham = Hamiltonian()
        energy, angles, history = run_vqe(
            np.array([0.2, -0.7, 0.3, 0.4]), ham, maxiter=300)
        self.assertLess(abs(energy - ham.exact_ground_energy()), 1e-6)
        self.assertGreater(len(history), 2)
        self.assertEqual(len(angles), 4)

    def test_noisy_optimization_and_sampling(self):
        ham = Hamiltonian()
        profile = [{'channel': 'depolarizing', 'p': 0.01},
                   {'channel': 'dephasing', 'p': 0.005}]
        theta0 = np.array([0.2, -0.7, 0.3, 0.4])
        exact, angles, history = run_vqe(theta0, ham, profile, maxiter=300)
        self.assertGreater(exact, ham.exact_ground_energy() + 0.01)
        self.assertLess(exact, -1.0)
        self.assertGreater(len(history), 2)
        sample = vqe_cost(angles, Circuit(2), ham, profile, shots=30000, n_averages=4)
        self.assertLess(abs(sample - exact), 0.02)


if __name__ == '__main__':
    unittest.main()
