import unittest
import torch
from src.train import token_statistics

class TokenStatisticsTests(unittest.TestCase):
    def test_scores_entropy_and_gradients(self):
        torch.manual_seed(29)
        logits=torch.randn(2,7,32,requires_grad=True)
        targets=torch.randint(0,32,(2,7))
        expected=logits.gather(-1,targets.unsqueeze(-1)).squeeze(-1)-torch.logsumexp(logits,-1)
        entropy=(torch.logsumexp(logits,-1)-(logits.softmax(-1)*logits).sum(-1)).mean()
        actual,observed=token_statistics(logits,targets)
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(observed,entropy)
        a=torch.autograd.grad(actual.sum(),logits,retain_graph=True)[0]
        b=torch.autograd.grad(expected.sum(),logits,retain_graph=True)[0]
        torch.testing.assert_close(a,b)
        skipped,zero=token_statistics(logits,targets,False)
        torch.testing.assert_close(skipped,expected)
        self.assertEqual(zero.item(),0)

if __name__=='__main__':unittest.main()

class CheckpointPolicyTests(unittest.TestCase):
    def test_short_and_long_sequence_switch(self):
        from unittest.mock import Mock
        from src.train import Experiment
        trainer=Experiment.__new__(Experiment)
        trainer.model=Mock()
        trainer.config={'checkpoint_free_max_tokens':1536}
        trainer.checkpointing_enabled=True
        trainer.configure_checkpointing(1536)
        trainer.model.gradient_checkpointing_disable.assert_called_once()
        trainer.configure_checkpointing(1000)
        trainer.model.gradient_checkpointing_disable.assert_called_once()
        trainer.configure_checkpointing(1537)
        trainer.model.gradient_checkpointing_enable.assert_called_once()
        self.assertTrue(trainer.checkpointing_enabled)
