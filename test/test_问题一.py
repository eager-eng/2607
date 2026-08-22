import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_PATH = PROJECT_ROOT / "code" / "问题一.py"
SPEC = importlib.util.spec_from_file_location("problem1", CODE_PATH)
problem1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(problem1)


class Problem1Tests(unittest.TestCase):
    def test_hydrogen_output_matches_rated_values(self):
        self.assertAlmostEqual(problem1.calculate_hydrogen_output(10.0, 0.70), 140.0)
        self.assertAlmostEqual(problem1.calculate_hydrogen_output(10.0, 0.80), 160.0)

    def test_grid_exchange_separates_buying_and_selling(self):
        load = np.array([20.0, 20.0, 20.0])
        generation = np.array([10.0, 20.0, 30.0])
        buy, sell = problem1.calculate_grid_exchange(load, generation)
        np.testing.assert_allclose(buy, [10.0, 0.0, 0.0])
        np.testing.assert_allclose(sell, [0.0, 0.0, 10.0])
        self.assertTrue(np.all(buy * sell == 0.0))

    def test_green_metrics_use_problem_statement_formula(self):
        metrics = problem1.calculate_green_metrics(
            load_energy_mwh=100.0,
            generation_energy_mwh=120.0,
            buy_energy_mwh=30.0,
            sell_energy_mwh=50.0,
        )
        self.assertAlmostEqual(metrics["新能源自发自用比例"], 1.0 / 6.0)
        self.assertAlmostEqual(metrics["新能源供电占比"], 0.70)
        self.assertAlmostEqual(metrics["上网电量比例"], 5.0 / 12.0)

    def test_daily_cost_reports_net_cost_only(self):
        costs = problem1.calculate_daily_cost(
            wind_energy_mwh=1.0,
            pv_energy_mwh=2.0,
            buy_energy_mwh=np.array([1.0]),
            buy_price_yuan_per_kwh=np.array([0.5]),
            sell_energy_mwh=np.array([0.25]),
            sell_price_yuan_per_kwh=np.array([0.4]),
            alk_energy_mwh=1.0,
            pem_energy_mwh=1.0,
            ammonia_energy_mwh=1.0,
            rated_hydrogen_demand_kg_per_h=300.0,
            ammonia_output_t=36.0,
        )
        expected = 150.0 + 240.0 + 500.0 - 100.0 + 100.0 + 150.0 + 2.0 + 60000.0 * 300.0 / 30.0 / 365.0
        self.assertAlmostEqual(costs["日净成本"], expected)
        self.assertAlmostEqual(costs["净吨氨成本"], expected / 36.0)
        self.assertNotIn("日毛成本", costs)
        self.assertNotIn("毛吨氨成本", costs)

    def test_actual_attachments_reproduce_energy_accounting(self):
        result = problem1.solve_problem1(PROJECT_ROOT)
        summary = result["summary"]
        checks = result["checks"]
        costs = result["costs"]
        self.assertAlmostEqual(summary["常规负荷电量"], 60.7200, places=4)
        self.assertAlmostEqual(summary["总负荷电量"], 558.7200, places=4)
        self.assertAlmostEqual(summary["风电电量"], 245.0480, places=4)
        self.assertAlmostEqual(summary["光伏电量"], 358.4000, places=4)
        self.assertAlmostEqual(summary["购电量"], 172.0438, places=4)
        self.assertAlmostEqual(summary["上网电量"], 216.7718, places=4)
        self.assertAlmostEqual(summary["氨产量"], 36.0, places=6)
        self.assertLess(abs(checks["电量平衡残差"]), 1e-9)
        self.assertLess(abs(checks["氢气平衡残差"]), 1e-9)
        self.assertAlmostEqual(costs["净吨氨成本"], 4367.999931012177)

    def test_single_entrypoint_exports_required_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "outputs"
            figure_dir = Path(temp_dir) / "figures"
            result = problem1.run_problem1(PROJECT_ROOT, output_dir, figure_dir)
            expected_outputs = {
                "问题一完整计算结果.json",
                "问题一逐时运行结果.csv",
                "问题一汇总结果.csv",
                "问题一绿电指标.csv",
                "问题一成本明细.csv",
                "问题一约束校验.csv",
            }
            self.assertTrue(expected_outputs.issubset({p.name for p in output_dir.iterdir()}))
            self.assertTrue((figure_dir / "问题一逐时功率平衡.pdf").stat().st_size > 1000)
            self.assertTrue((figure_dir / "问题一逐时功率平衡.png").stat().st_size > 1000)
            self.assertAlmostEqual(result["costs"]["净吨氨成本"], 4367.999931012177)
            cost_text = (output_dir / "问题一成本明细.csv").read_text(encoding="utf-8-sig")
            self.assertNotIn("毛成本", cost_text)


if __name__ == "__main__":
    unittest.main()
