"""Tests for the MemX CLI."""

from click.testing import CliRunner
from memx.cli import cli


class TestCLI:
    def setup_method(self):
        self.runner = CliRunner()

    def test_version(self):
        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output or "version" in result.output.lower()

    def test_demo(self):
        result = self.runner.invoke(cli, ["demo"])
        assert result.exit_code == 0
        assert "MemX" in result.output
        assert "rag" in result.output.lower() or "Adding" in result.output

    def test_benchmark(self):
        result = self.runner.invoke(cli, ["benchmark", "--n", "50"])
        assert result.exit_code == 0
        assert "Benchmark" in result.output
        assert "ms" in result.output

    def test_init(self):
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert "initialized" in result.output.lower()

    def test_stats(self):
        result = self.runner.invoke(cli, ["stats"])
        assert result.exit_code == 0
        assert "Stats" in result.output
