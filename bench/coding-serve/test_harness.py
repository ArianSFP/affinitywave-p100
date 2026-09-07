import io
import json
from pathlib import Path
import signal
import subprocess
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

import run
import server_probe
import compare_boundaries
import compare_quality


class HarnessTests(unittest.TestCase):
    def test_qualified_environment_keeps_original_path(self):
        env = run.environment(Path('/example/build'), 8128, 'qualified')
        self.assertNotIn('GGML_CUDA_AW_SERVE', env)
        self.assertEqual(env['GGML_CUDA_AW_DIAGONAL_SERVICE'], 'panel2048')
        self.assertEqual(env['GGML_CUDA_AW_P100_EXACT'], '1')
        self.assertEqual(env['GGML_CUDA_AW_WAVE_TOKENS'], '8128')
        self.assertNotIn('PYTHONPATH', env)

    def test_serving_is_explicit(self):
        env = run.environment(Path('/example/build'), 8128, 'serve')
        self.assertEqual(env['GGML_CUDA_AW_SERVE'], '1')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0,1,2,3')

    def test_native_control_has_no_aw_layout(self):
        env = run.environment(Path('/example/build'), 512, 'normal')
        self.assertFalse(any(k.startswith('GGML_CUDA_AW_') for k in env))
        self.assertNotIn('GGML_CUDA_AFFINITY_WAVE', env)
        self.assertEqual(env['GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F'], '1')

    @patch('run.os.killpg')
    def test_stop_only_targets_owned_process_group(self, killpg):
        child = Mock(pid=12345)
        child.poll.return_value = None
        run.stop(child)
        killpg.assert_called_once_with(12345, signal.SIGTERM)

    @patch('run.os.killpg')
    def test_stop_escalates_only_owned_group(self, killpg):
        child = Mock(pid=12345)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('test', 10), 0]
        run.stop(child)
        self.assertEqual(killpg.call_args_list[-1].args, (12345, signal.SIGKILL))

    @patch('run.os.killpg')
    def test_finished_child_is_not_signalled(self, killpg):
        child = Mock()
        child.poll.return_value = 0
        run.stop(child)
        killpg.assert_not_called()

    def test_stream_records_real_tokens_and_final_timings(self):
        events = [dict(content='int', tokens=[42], stop=False),
                  dict(content='', tokens=[], stop=True,
                       timings=dict(cache_n=128, prompt_n=65, prompt_ms=20))]
        body = b': keepalive\n\n' + b''.join(b'data: ' + json.dumps(e).encode() + b'\n\n' for e in events)
        body += b'data: [DONE]\n'
        with patch('server_probe.request', return_value=io.BytesIO(body)):
            result = server_probe.completion('http://127.0.0.1', [1, 2], True)
        self.assertEqual(result['tokens'], [42])
        self.assertEqual(result['content'], 'int')
        self.assertEqual(result['timings']['prompt_n'], 65)
        self.assertIsNotNone(result['ttft_ms'])

    def test_missing_final_event_fails(self):
        with patch('server_probe.request', return_value=io.BytesIO(b'data: {"content":"x"}\n')):
            with self.assertRaisesRegex(RuntimeError, 'final timings'):
                server_probe.completion('http://127.0.0.1', [1], False)

    def test_api_error_fails(self):
        with patch('server_probe.request', return_value=io.BytesIO(b'data: {"error":"test"}\n')):
            with self.assertRaisesRegex(RuntimeError, 'test'):
                server_probe.completion('http://127.0.0.1', [1], False)

    def test_boundary_comparison_checks_every_logit(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory)/'left', Path(directory)/'right'
            header = struct.pack('=I', 1) + b'a' + struct.pack('=I', 3)
            left.write_bytes(header + struct.pack('=fff', 1, 2, 3))
            right.write_bytes(left.read_bytes())
            self.assertTrue(compare_boundaries.compare(left, right)['byte_identical'])
            right.write_bytes(header + struct.pack('=fff', 1, 2, 4))
            result = compare_boundaries.compare(left, right)
            self.assertFalse(result['byte_identical'])
            self.assertEqual(result['mismatches'][0]['changed_logits'], 1)

    def test_boundary_comparison_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'truncated'
            path.write_bytes(struct.pack('=I', 1) + b'a' + struct.pack('=I', 2) + b'x')
            with self.assertRaisesRegex(ValueError, 'truncated logits'):
                list(compare_boundaries.records(path))

    def test_quality_comparison_uses_paired_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory)/'left', Path(directory)/'right'
            header = b'AWQLOG01' + struct.pack('=III', 16, 1, 2)
            left.write_bytes(header + struct.pack('=Iff', 0, 0, 0))
            right.write_bytes(left.read_bytes())
            result = compare_quality.compare(left, right)
            self.assertTrue(result['byte_identical'])
            self.assertTrue(result['ppl_gate_pass'])
            self.assertAlmostEqual(result['base_ppl'], 2)
            right.write_bytes(header + struct.pack('=Iff', 0, 0, 1))
            self.assertFalse(compare_quality.compare(left, right)['ppl_gate_pass'])
            right.write_bytes(header + struct.pack('=Iff', 1, 0, 0))
            with self.assertRaisesRegex(ValueError, 'targets differ'):
                compare_quality.compare(left, right)


if __name__ == '__main__':
    unittest.main()
