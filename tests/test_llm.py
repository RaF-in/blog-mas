"""Tests for the LLM factory (create_llm)."""

from unittest.mock import patch

from blog_mas.llm import create_llm


def _patch_env():
    return patch.dict("os.environ", {}, clear=True)


class TestCreateLlmDefaults:
    @patch("blog_mas.llm.ChatHuggingFace")
    @patch("blog_mas.llm.HuggingFaceEndpoint")
    def test_creates_with_default_model(self, mock_endpoint, mock_chat):
        with _patch_env():
            create_llm()
        mock_endpoint.assert_called_once_with(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            task="text-generation",
            temperature=0.3,
            huggingfacehub_api_token=None,
        )
        mock_chat.assert_called_once_with(llm=mock_endpoint.return_value)

    @patch("blog_mas.llm.ChatHuggingFace")
    @patch("blog_mas.llm.HuggingFaceEndpoint")
    def test_creates_with_custom_model(self, mock_endpoint, mock_chat):
        with _patch_env():
            create_llm(model="meta-llama/Llama-3-8B")
        mock_endpoint.assert_called_once_with(
            repo_id="meta-llama/Llama-3-8B",
            task="text-generation",
            temperature=0.3,
            huggingfacehub_api_token=None,
        )

    @patch("blog_mas.llm.ChatHuggingFace")
    @patch("blog_mas.llm.HuggingFaceEndpoint")
    def test_forwards_temperature(self, mock_endpoint, mock_chat):
        with _patch_env():
            create_llm(temperature=0.7)
        mock_endpoint.assert_called_once_with(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            task="text-generation",
            temperature=0.7,
            huggingfacehub_api_token=None,
        )


class TestCreateLlmApiKey:
    @patch("blog_mas.llm.ChatHuggingFace")
    @patch("blog_mas.llm.HuggingFaceEndpoint")
    def test_uses_explicit_api_key(self, mock_endpoint, mock_chat):
        create_llm(api_key="hf_test_key_123")
        mock_endpoint.assert_called_once_with(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            task="text-generation",
            temperature=0.3,
            huggingfacehub_api_token="hf_test_key_123",
        )

    @patch("blog_mas.llm.ChatHuggingFace")
    @patch("blog_mas.llm.HuggingFaceEndpoint")
    def test_falls_back_to_env_var(self, mock_endpoint, mock_chat):
        with patch.dict("os.environ", {"HF_TOKEN": "hf_env_key_456"}):
            create_llm()
        mock_endpoint.assert_called_once_with(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            task="text-generation",
            temperature=0.3,
            huggingfacehub_api_token="hf_env_key_456",
        )

    @patch("blog_mas.llm.ChatHuggingFace")
    @patch("blog_mas.llm.HuggingFaceEndpoint")
    def test_prefers_explicit_key_over_env(self, mock_endpoint, mock_chat):
        with patch.dict("os.environ", {"HF_TOKEN": "hf_env_key"}):
            create_llm(api_key="hf_explicit_key")
        mock_endpoint.assert_called_once_with(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            task="text-generation",
            temperature=0.3,
            huggingfacehub_api_token="hf_explicit_key",
        )

    @patch("blog_mas.llm.ChatHuggingFace")
    @patch("blog_mas.llm.HuggingFaceEndpoint")
    def test_returns_none_when_no_key(self, mock_endpoint, mock_chat):
        with _patch_env():
            create_llm()
        mock_endpoint.assert_called_once_with(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            task="text-generation",
            temperature=0.3,
            huggingfacehub_api_token=None,
        )
