import unittest
from unittest.mock import patch, MagicMock
from llm_provider import create_client, validate_endpoint, ProviderError

class ProviderTests(unittest.TestCase):
    def test_request_formats(self):
        for provider in ('OpenAI', 'Anthropic', 'Google Gemini'):
            with patch('llm_provider.httpx.Client') as factory:
                response = MagicMock(status_code=200)
                response.json.return_value = ({'content':[{'type':'text','text':'{"answerable":true}'}]}
                    if provider == 'Anthropic' else {'choices':[{'message':{'content':'{"answerable":true}'}}]})
                post = factory.return_value.__enter__.return_value.post
                post.return_value = response
                result = create_client(provider, 'test-model', 'test-key').invoke([('system','Rules'),('human','Question')])
                self.assertEqual(result.content, '{"answerable":true}')
                args, kwargs = post.call_args
                self.assertEqual(kwargs['json']['model'], 'test-model')
                self.assertEqual(kwargs['json']['messages'][-1]['role'], 'user')
                if provider == 'Anthropic':
                    self.assertEqual(kwargs['json']['system'], 'Rules')
                    self.assertEqual(kwargs['headers']['x-api-key'], 'test-key')
                    self.assertTrue(args[0].endswith('/messages'))
                else:
                    self.assertEqual(kwargs['headers']['Authorization'], 'Bearer test-key')
                    self.assertTrue(args[0].endswith('/chat/completions'))
                self.assertFalse(factory.call_args.kwargs['follow_redirects'])

    def test_custom_host_requires_operator_configuration(self):
        with patch.dict('os.environ', {}, clear=True):
            for url in ['http://localhost:8000/v1','https://unknown.example/v1',
                        'https://api.openai.com@evil.test/v1','https://api.openai.com/v1?key=secret']:
                with self.assertRaises(ValueError): validate_endpoint(url)
        with patch.dict('os.environ', {'RAG_ALLOWED_LLM_HOSTS':'models.example'}):
            self.assertEqual(validate_endpoint('https://models.example/v1/'),'https://models.example/v1')

    def test_status_errors_preserve_retry_status_without_body(self):
        with patch('llm_provider.httpx.Client') as factory:
            factory.return_value.__enter__.return_value.post.return_value = MagicMock(
                status_code=429, headers={'retry-after':'5'}, text='secret body')
            with self.assertRaises(ProviderError) as error:
                create_client('OpenAI', 'model', 'key').invoke([('human','Question')])
            self.assertEqual(error.exception.status_code,429)
            self.assertNotIn('secret',str(error.exception))

    def test_missing_models_rejected_before_network(self):
        with self.assertRaises(ValueError): create_client('OpenAI','','key')
