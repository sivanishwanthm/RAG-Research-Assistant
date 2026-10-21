import threading
import unittest
from unittest.mock import patch
from app_backend import ResearchService


class CredentialTests(unittest.TestCase):
    def test_personal_keys_are_request_scoped_and_blank_does_not_fallback(self):
        service = ResearchService.__new__(ResearchService)
        service.lock = threading.RLock()
        service.search = lambda question: []
        with patch.dict('os.environ', {'GROQ_API_KEY':'server-test-key'}), \
             patch('langchain_groq.ChatGroq') as client, \
             patch('app_backend.grounded_answer', return_value='Answer'):
            first = service.ask('Question', api_key='user-one-test-key')
            service.ask('Question', api_key='user-two-test-key')
            service.ask('Question')
            self.assertEqual([c.kwargs['groq_api_key'] for c in client.call_args_list],
                             ['user-one-test-key','user-two-test-key','server-test-key'])
            with self.assertRaises(ValueError): service.ask('Question', api_key=' ')
            self.assertEqual(client.call_count, 3)
            self.assertNotIn('user-one-test-key', str(first))
            self.assertNotIn('user-one-test-key', str(service.__dict__))
