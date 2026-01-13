import unittest

from rewoo_agent import ReWOOAgent
from tests.DummyData import patient_x


class DummyLLM:
    def __call__(self, *args, **kwargs):
        return ""


class TestReWOOAgent(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.agent = ReWOOAgent(llm=DummyLLM())
        self.agent.patient = patient_x

    def test_parse_planner_output_variants(self):
        text = """Plan: Check PE
#e1: Physical Exam[]
Plan- Order labs
#E2 = Labs[WBC, CRP]
Plan: imaging
#E3 = Imaging[region=Abdomen, modality=CT]
"""
        plans, assignments = self.agent._parse_planner_output(text)
        self.assertEqual(
            plans,
            ["Plan: Check PE", "Plan: Order labs", "Plan: imaging"],
        )
        self.assertEqual(
            assignments,
            [
                ("#E1", "Physical Exam[]"),
                ("#E2", "Labs[WBC, CRP]"),
                ("#E3", "Imaging[region=Abdomen, modality=CT]"),
            ],
        )

    def test_execute_resolves_inputs(self):
        assignments = [("#E1", "Imaging[region=Abdomen, modality=CT]")]
        evidences, resolved_inputs = self.agent._execute(assignments)
        self.assertIn("#E1", evidences)
        self.assertEqual(resolved_inputs.get("#E1"), "region=Abdomen, modality=CT")
        self.assertIn("Imaging:\nAbdomen CT:", evidences["#E1"])

    def test_build_worker_log_with_plans(self):
        plans = ["Plan: Check PE", "Plan: Imaging"]
        assignments = [
            ("#E1", "Physical Examination[]"),
            ("#E2", "Imaging[region=Abdomen, modality=CT]"),
        ]
        evidences = {
            "#E1": "Physical Examination:\nNormal exam.\n",
            "#E2": "Imaging:\nAbdomen CT: Appendicitis.\n",
        }
        resolved_inputs = {"#E2": "region=Abdomen, modality=CT"}
        log = self.agent._build_worker_log(plans, assignments, evidences, resolved_inputs)
        self.assertIn("Plan: Check PE", log)
        self.assertIn("#E2 = Imaging[region=Abdomen, modality=CT]", log)
        self.assertIn("Evidence:\nImaging:\nAbdomen CT: Appendicitis.", log)

    def test_build_worker_log_no_plan(self):
        plans = []
        assignments = [("#E1", "Physical Examination[]")]
        evidences = {"#E1": "Physical Examination:\nOK.\n"}
        resolved_inputs = {}
        log = self.agent._build_worker_log(plans, assignments, evidences, resolved_inputs)
        self.assertIn("Plan: (no plan provided)", log)
        self.assertIn("#E1 = Physical Examination[]", log)
        self.assertIn("Evidence:\nPhysical Examination:\nOK.", log)

    def test_tool_alias_physical_exam(self):
        obs = self.agent._run_tool("Physical Exam", "")
        expected = f"Physical Examination:\n{patient_x['Physical Examination']}\n"
        self.assertEqual(obs, expected)


if __name__ == "__main__":
    unittest.main()
