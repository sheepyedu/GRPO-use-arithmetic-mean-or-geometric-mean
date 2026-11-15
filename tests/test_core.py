import unittest
from fractions import Fraction

from src.data import PromptSchedule, canonical, exclude_duplicates
from src.rewards import parse_answer, score


class RewardTests(unittest.TestCase):
    def test_strict_final(self):
        self.assertEqual(
            parse_answer("<think>reason</think>\nFinal answer: 1,200", "gsm8k"),
            Fraction(1200),
        )
        self.assertEqual(parse_answer("Final answer: 3/2", "gsm8k"), Fraction(3, 2))
        for text in (
            "12",
            "Final answer: 2 or 3",
            "Final answer: 1\nFinal answer: 2",
            "<think>Final answer: 1",
            "Final answer: 1\nextra",
        ):
            self.assertIsNone(parse_answer(text, "gsm8k"))
        self.assertIsNone(parse_answer("Final answer: A", "arc", ["1", "2"]))
        self.assertEqual(parse_answer("Final answer: YES", "boolq"), "yes")
        self.assertIsNone(parse_answer("Final answer: yes/no", "boolq"))
        self.assertIsNone(parse_answer("Final answer: 1", "gsm8k", truncated=True))
        self.assertEqual(
            parse_answer("Final answer: 1\n", "gsm8k", truncated=True), Fraction(1)
        )
        self.assertEqual(score("Final answer: 0", {"task": "gsm8k", "answer": "0"}), 1)


class DataTests(unittest.TestCase):
    def test_digits_preserved(self):
        self.assertNotEqual(canonical("buy 1 item"), canonical("buy 2 items"))

    def test_duplicates(self):
        protected = [{"question": "A has 12 apples."}]
        rows = [{"question": "A HAS 12 apples!"}, {"question": "A has 13 apples."}]
        kept, audit = exclude_duplicates(rows, protected)
        self.assertEqual(len(kept), 1)
        self.assertEqual(audit["exact"], 1)

    def test_paired_schedule(self):
        rows = [
            {"task": t, "id": f"{t}{i}"}
            for t in ("gsm8k", "arc", "boolq")
            for i in range(100)
        ]
        a, b = PromptSchedule(rows, "mixed", 17), PromptSchedule(rows, "mixed", 17)
        for _ in range(10):
            batch = a.batch()
            self.assertEqual(batch, b.batch())
            self.assertEqual(
                [sum(r["task"] == t for r in batch) for t in ("gsm8k", "arc", "boolq")],
                [32, 16, 16],
            )


try:
    import torch

    from src.objectives import group_advantages, sequence_log_ratio, sequence_loss
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch required for objective tests")
class ObjectiveTests(unittest.TestCase):
    def test_extreme_log_domain(self):
        d = torch.tensor([[1000.0, -1000.0]])
        mask = torch.ones_like(d, dtype=torch.bool)
        self.assertTrue(torch.isfinite(sequence_log_ratio(d, mask, "am")).all())
        self.assertTrue(torch.isfinite(sequence_log_ratio(d, mask, "gm")).all())

    def test_inequality_and_mask(self):
        delta = torch.tensor([[1.0, -2.0, 999.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        am = sequence_log_ratio(delta, mask, "am")
        gm = sequence_log_ratio(delta, mask, "gm")
        self.assertTrue(torch.all(am >= gm - 1e-12))
        self.assertTrue(torch.isfinite(am).all())
        self.assertEqual(am[1].item(), 0)

    def test_identity_gradients(self):
        derivatives = []
        for mode in ("am", "gm"):
            d = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
            s = sequence_log_ratio(d, torch.ones_like(d, dtype=torch.bool), mode).exp()
            s.sum().backward()
            derivatives.append(d.grad)
        torch.testing.assert_close(*derivatives)

    def test_clipping_and_zero_rewards(self):
        advantages, std = group_advantages(torch.ones(16))
        self.assertTrue(torch.all(advantages == 0))
        current = torch.log(torch.tensor([[1.5], [0.5], [0.5], [1.5]]))
        zero = torch.zeros_like(current)
        a = torch.tensor([1.0, 1.0, -1.0, -1.0])
        for mode in ("am", "gm"):
            loss, _ = sequence_loss(
                current,
                zero,
                zero,
                torch.ones_like(current, dtype=torch.bool),
                a,
                mode,
                beta=0,
            )
            expected = -torch.tensor([1.2, 0.5, -0.8, -1.5]).mean()
            torch.testing.assert_close(loss, expected)

    def test_tiny_qwen_update(self):
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3Config, Qwen3ForCausalLM

        from src.train import token_scores

        torch.manual_seed(17)
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=256,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
            )
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.eval()
        ids = torch.tensor([[2, 3, 4, 5, 6]])
        with torch.no_grad():
            old, _ = token_scores(model, ids, 2)
        frozen = old.clone()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.01
        )
        for _ in range(2):
            model.train()
            current, _ = token_scores(model, ids, 2)
            loss, _ = sequence_loss(
                current,
                old,
                old,
                torch.ones_like(old, dtype=torch.bool),
                torch.ones(1),
                "am",
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.testing.assert_close(old, frozen)
        self.assertFalse(torch.equal(token_scores(model, ids, 2)[0], old))


if __name__ == "__main__":
    unittest.main()
